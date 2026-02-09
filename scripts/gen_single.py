#!/usr/bin/env python3
"""Generate single precision (float) LAPACK C source files from double precision.

Usage:
    python scripts/gen_single.py              # convert all files
    python scripts/gen_single.py --dry-run    # preview without writing
    python scripts/gen_single.py --file dgetrf.c           # single file
    python scripts/gen_single.py --file dgetrf.c --dry-run # preview single file

Reads src/d/*.c, applies token-aware precision substitutions, writes src/s/*.c.
Comments are left unchanged except for @file tags and uppercase routine names.
String literals only get uppercase routine name substitutions (for xerbla).
"""

import os
import re
import sys
import shutil
import argparse
import tempfile

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_D = os.path.join(PROJECT_ROOT, "src", "d")
SRC_S = os.path.join(PROJECT_ROOT, "src", "s")
DOUBLE_HEADER = os.path.join(PROJECT_ROOT, "src", "include",
                             "semicolon_lapack_double.h")

# Files to skip (mixed-precision routines with distinct logic)
EXCLUDE_FILES = {
    "dsgesv.c",   # mixed-precision iterative refinement (general)
    "dsposv.c",   # mixed-precision iterative refinement (SPD)
    "dlag2s.c",   # double -> single conversion
    "dlat2s.c",   # double -> single symmetric conversion
}

# Precision-independent files that live in src/auxiliary/, not src/d/ or src/s/
AUXILIARY_FILES = {
    "ieeeck.c",
    "ilaenv2stage.c",
    "iparam2stage.c",
}

# Hand-written files in src/s/ that are NOT generated from src/d/.
# The script preserves these across full runs.
PRESERVE_FILES = [
    "slag2d.c",   # single -> double conversion (distinct logic)
]

# Files that need manual review after generation
REVIEW_KEYWORDS = ["ldexp"]

# Identifiers with embedded precision marker (not a simple d -> s prefix)
EMBEDDED_PRECISION = {
    "iladlc": "ilaslc",
    "iladlr": "ilaslr",
}


# ---------------------------------------------------------------------------
# Extract LAPACK routine names from double header
# ---------------------------------------------------------------------------
def extract_lapack_names(header_path):
    """Return set of all d-prefixed identifiers declared in the header."""
    names = set()
    with open(header_path) as f:
        for line in f:
            m = re.search(r"SEMICOLON_API\s+\w[\w*\s]*\s+(d\w+)\s*\(", line)
            if m:
                names.add(m.group(1))
    # callback typedefs
    names.add("dselect2_t")
    names.add("dselect3_t")
    return names


# ---------------------------------------------------------------------------
# Build substitution rule lists
# ---------------------------------------------------------------------------
def build_code_subs(lapack_names):
    """Ordered (pattern, replacement) pairs for CODE segments."""
    subs = []

    # --- 1. Header include (inside #include "...") ---
    subs.append((re.compile(r"semicolon_lapack_double\.h"),
                 "semicolon_lapack_single.h"))

    # --- 2. CBLAS special: cblas_idamax -> cblas_isamax ---
    subs.append((re.compile(r"\bcblas_idamax\b"), "cblas_isamax"))

    # --- 3. CBLAS general: cblas_d* -> cblas_s* ---
    subs.append((re.compile(r"\bcblas_d(\w+)\b"), r"cblas_s\1"))

    # --- 4. ILA embedded precision ---
    subs.append((re.compile(r"\biladlc\b"), "ilaslc"))
    subs.append((re.compile(r"\biladlr\b"), "ilaslr"))

    # --- 5. float.h constants ---
    subs.append((re.compile(r"\bDBL_EPSILON\b"), "FLT_EPSILON"))
    subs.append((re.compile(r"\bDBL_MIN_EXP\b"), "FLT_MIN_EXP"))
    subs.append((re.compile(r"\bDBL_MAX_EXP\b"), "FLT_MAX_EXP"))
    subs.append((re.compile(r"\bDBL_MANT_DIG\b"), "FLT_MANT_DIG"))
    subs.append((re.compile(r"\bDBL_MIN\b"), "FLT_MIN"))
    subs.append((re.compile(r"\bDBL_MAX\b"), "FLT_MAX"))

    # --- 6. LAPACK routine names (longest first to avoid partial matches) ---
    for name in sorted(lapack_names, key=len, reverse=True):
        sname = "s" + name[1:]
        subs.append((re.compile(r"\b" + re.escape(name) + r"\b"), sname))

    # --- 7. Type keyword ---
    subs.append((re.compile(r"\bdouble\b"), "float"))

    # --- 8. Math functions (word-boundary; order matters for substrings) ---
    math_funcs = [
        ("copysign", "copysignf"),
        ("log10",    "log10f"),
        ("atan2",    "atan2f"),
        ("ldexp",    "ldexpf"),
        ("floor",    "floorf"),
        ("round",    "roundf"),
        ("sqrt",     "sqrtf"),
        ("fabs",     "fabsf"),
        ("fmax",     "fmaxf"),
        ("fmin",     "fminf"),
        ("ceil",     "ceilf"),
        ("pow",      "powf"),
        ("log",      "logf"),
        ("tan",      "tanf"),
        ("sin",      "sinf"),
        ("cos",      "cosf"),
    ]
    for old, new in math_funcs:
        subs.append((re.compile(r"\b" + re.escape(old) + r"\b"), new))

    # --- 9. Float literals: 1.0 -> 1.0f  (not already suffixed) ---
    # Matches: digits.digits, optionally with exponent, NOT followed by f
    subs.append((re.compile(r"(\d+\.\d+(?:[eE][+-]?\d+)?)(?!f\b)"),
                 r"\1f"))

    return subs


def build_string_subs(lapack_names):
    """Substitutions for string literals (uppercase xerbla names + header)."""
    subs = []
    # Header include appears inside "..." in #include directives
    subs.append((re.compile(r"semicolon_lapack_double\.h"),
                 "semicolon_lapack_single.h"))
    # Embedded precision identifiers (before general names)
    for old, new in EMBEDDED_PRECISION.items():
        subs.append((re.compile(re.escape(old.upper())), new.upper()))
    # Uppercase routine names: "DGETRF" -> "SGETRF"
    for name in sorted(lapack_names, key=len, reverse=True):
        upper_d = name.upper()
        upper_s = ("S" + name[1:]).upper()
        subs.append((re.compile(re.escape(upper_d)), upper_s))
    return subs


def build_comment_subs(lapack_names):
    """Substitutions for comment segments (@file tags, routine names)."""
    subs = []
    # @file renames (d-prefix and embedded-precision files)
    subs.append((re.compile(r"(@file\s+)d(\w+\.c)"), r"\1s\2"))
    for old_f, new_f in FILENAME_RENAMES.items():
        subs.append((re.compile(r"(@file\s+)" + re.escape(old_f)),
                     r"\1" + new_f))
    # float.h constants in comments
    subs.append((re.compile(r"\bDBL_MIN_EXP\b"), "FLT_MIN_EXP"))
    subs.append((re.compile(r"\bDBL_MAX_EXP\b"), "FLT_MAX_EXP"))
    subs.append((re.compile(r"\bDBL_MANT_DIG\b"), "FLT_MANT_DIG"))
    # Embedded precision identifiers (before general LAPACK names)
    for old, new in EMBEDDED_PRECISION.items():
        subs.append((re.compile(r"\b" + re.escape(old.upper()) + r"\b"),
                     new.upper()))
        subs.append((re.compile(r"\b" + re.escape(old) + r"\b"), new))
    # Routine names in both cases: DGETRF -> SGETRF, dgetrf -> sgetrf
    for name in sorted(lapack_names, key=len, reverse=True):
        sname = "s" + name[1:]
        # uppercase
        subs.append((re.compile(r"\b" + re.escape(name.upper()) + r"\b"),
                     sname.upper()))
        # lowercase (the actual function name as used in :c:func:`dgetrf` etc.)
        subs.append((re.compile(r"\b" + re.escape(name) + r"\b"), sname))
    return subs


# ---------------------------------------------------------------------------
# Tokenizer: split lines into code / comment / string segments
# ---------------------------------------------------------------------------
def tokenize_line(line):
    """Split a line into tagged segments.

    Returns list of (tag, text) where tag is one of:
        'code'          - C source code
        'line_comment'  - // ... to end of line
        'block_comment' - /* ... */ (complete on this line)
        'block_start'   - /* ... (continues to next line)
        'string'        - "..."
        'char'          - '...'
    """
    segments = []
    buf = []
    i = 0
    n = len(line)

    while i < n:
        # --- // line comment ---
        if line[i] == "/" and i + 1 < n and line[i + 1] == "/":
            if buf:
                segments.append(("code", "".join(buf)))
                buf = []
            segments.append(("line_comment", line[i:]))
            return segments

        # --- /* block comment --- */
        if line[i] == "/" and i + 1 < n and line[i + 1] == "*":
            if buf:
                segments.append(("code", "".join(buf)))
                buf = []
            end = line.find("*/", i + 2)
            if end >= 0:
                segments.append(("block_comment", line[i : end + 2]))
                i = end + 2
            else:
                segments.append(("block_start", line[i:]))
                return segments
            continue

        # --- string literal ---
        if line[i] == '"':
            if buf:
                segments.append(("code", "".join(buf)))
                buf = []
            j = i + 1
            while j < n:
                if line[j] == "\\":
                    j += 2
                elif line[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            segments.append(("string", line[i:j]))
            i = j
            continue

        # --- char literal ---
        if line[i] == "'":
            if buf:
                segments.append(("code", "".join(buf)))
                buf = []
            j = i + 1
            while j < n:
                if line[j] == "\\":
                    j += 2
                elif line[j] == "'":
                    j += 1
                    break
                else:
                    j += 1
            segments.append(("char", line[i:j]))
            i = j
            continue

        buf.append(line[i])
        i += 1

    if buf:
        segments.append(("code", "".join(buf)))

    return segments


def apply_subs(text, subs):
    """Apply a list of (compiled_regex, replacement) pairs to text."""
    for pattern, repl in subs:
        text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# Process a single file
# ---------------------------------------------------------------------------
def process_segments(text, code_subs, string_subs, comment_subs):
    """Tokenize a text fragment and apply appropriate substitutions."""
    segments = tokenize_line(text)
    parts = []
    block_started = False
    for tag, seg in segments:
        if tag == "code":
            parts.append(apply_subs(seg, code_subs))
        elif tag == "string":
            parts.append(apply_subs(seg, string_subs))
        elif tag in ("line_comment", "block_comment"):
            parts.append(apply_subs(seg, comment_subs))
        elif tag == "block_start":
            parts.append(apply_subs(seg, comment_subs))
            block_started = True
        else:  # char
            parts.append(seg)
    return "".join(parts), block_started


def process_file(src_path, code_subs, string_subs, comment_subs):
    """Read a double-precision file and return single-precision content."""
    with open(src_path) as f:
        lines = f.readlines()

    output = []
    in_block = False

    for line in lines:
        if in_block:
            end = line.find("*/")
            if end >= 0:
                # comment part (up to and including */)
                comment_part = apply_subs(line[: end + 2], comment_subs)
                # rest of line is code again
                rest, in_block = process_segments(
                    line[end + 2 :], code_subs, string_subs, comment_subs
                )
                output.append(comment_part + rest)
            else:
                # still inside block comment
                output.append(apply_subs(line, comment_subs))
        else:
            result, in_block = process_segments(
                line, code_subs, string_subs, comment_subs
            )
            output.append(result)

    return "".join(output)


# Explicit filename renames for files where the precision marker is embedded
FILENAME_RENAMES = {
    "iladlc.c": "ilaslc.c",
    "iladlr.c": "ilaslr.c",
}


def rename_file(name):
    """dXXX.c -> sXXX.c, with special cases for embedded precision markers."""
    if name in FILENAME_RENAMES:
        return FILENAME_RENAMES[name]
    if name.startswith("d"):
        return "s" + name[1:]
    return name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate single-precision LAPACK C files from double"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print converted output instead of writing files"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Process only this file (basename, e.g. dgetrf.c)"
    )
    args = parser.parse_args()

    # Extract routine names from the double header
    print(f"Reading LAPACK names from {DOUBLE_HEADER} ...", file=sys.stderr)
    lapack_names = extract_lapack_names(DOUBLE_HEADER)
    print(f"  Found {len(lapack_names)} d-prefixed identifiers", file=sys.stderr)

    # Build substitution tables
    code_subs = build_code_subs(lapack_names)
    string_subs = build_string_subs(lapack_names)
    comment_subs = build_comment_subs(lapack_names)

    # Collect source files
    if args.file:
        src_files = [args.file]
    else:
        src_files = sorted(f for f in os.listdir(SRC_D) if f.endswith(".c"))

    # Back up hand-written files, then clean src/s/ for a fresh generation
    preserved = {}
    if not args.dry_run and not args.file:
        for pf in PRESERVE_FILES:
            pf_path = os.path.join(SRC_S, pf)
            if os.path.exists(pf_path):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".c")
                shutil.copy2(pf_path, tmp.name)
                preserved[pf] = tmp.name
                tmp.close()
        # Remove all .c files in src/s/ for a clean slate
        if os.path.isdir(SRC_S):
            for f in os.listdir(SRC_S):
                if f.endswith(".c"):
                    os.remove(os.path.join(SRC_S, f))

    converted = 0
    skipped = 0
    needs_review = []

    for fname in src_files:
        if fname in EXCLUDE_FILES:
            print(f"  SKIP (mixed-precision): {fname}", file=sys.stderr)
            skipped += 1
            continue
        if fname in AUXILIARY_FILES:
            print(f"  SKIP (precision-independent): {fname}", file=sys.stderr)
            skipped += 1
            continue

        src_path = os.path.join(SRC_D, fname)
        dst_name = rename_file(fname)
        dst_path = os.path.join(SRC_S, dst_name)

        content = process_file(src_path, code_subs, string_subs, comment_subs)

        # Check if file needs manual review
        with open(src_path) as f:
            src_text = f.read()
        for kw in REVIEW_KEYWORDS:
            if kw in src_text:
                needs_review.append((fname, kw))

        if args.dry_run:
            print(f"=== {fname} -> {dst_name} ===", file=sys.stderr)
            print(content)
        else:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            with open(dst_path, "w") as f:
                f.write(content)
            converted += 1

    # Restore hand-written files
    if not args.dry_run:
        for pf, tmp_path in preserved.items():
            dst = os.path.join(SRC_S, pf)
            shutil.copy2(tmp_path, dst)
            os.unlink(tmp_path)
            print(f"  Restored preserved file: {pf}", file=sys.stderr)

        print(f"\nConverted {converted} files, skipped {skipped}")
        print(f"Output: {SRC_S}/")

    if needs_review:
        print(f"\nFiles needing manual review ({len(needs_review)}):")
        for fname, kw in needs_review:
            print(f"  {fname} -- contains '{kw}' (precision-dependent constant)")


if __name__ == "__main__":
    main()
