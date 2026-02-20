#!/usr/bin/env python3
"""Generate single precision (float) LAPACK test files from double precision.

Usage:
    python scripts/gen_single_tests.py              # convert all files
    python scripts/gen_single_tests.py --dry-run    # preview without writing
    python scripts/gen_single_tests.py --file dget01.c           # single file (testutils)
    python scripts/gen_single_tests.py --file test_dchkge.c      # single file (driver)
    python scripts/gen_single_tests.py --file test_dchkge.c --dry-run

Reads tests/d/ tree, applies token-aware precision substitutions, writes tests/s/ tree.
Comments are left unchanged except for @file tags and uppercase routine names.
String literals only get uppercase routine name substitutions (for xerbla) and
precision-path substitutions ("DGE" -> "SGE").
"""

import os
import re
import sys
import argparse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TESTS_D = os.path.join(PROJECT_ROOT, "tests", "d")
TESTS_S = os.path.join(PROJECT_ROOT, "tests", "s")
TESTUTILS_D = os.path.join(TESTS_D, "testutils")
TESTUTILS_S = os.path.join(TESTS_S, "testutils")
SMOKE_D = os.path.join(TESTS_D, "smoke")
SMOKE_S = os.path.join(TESTS_S, "smoke")
DOUBLE_HEADER = os.path.join(PROJECT_ROOT, "src", "include",
                             "semicolon_lapack_double.h")
VERIFY_HEADER = os.path.join(TESTUTILS_D, "verify.h")

# Files to skip (mixed-precision tests with no single-precision analogue)
EXCLUDE_FILES = {
    "test_ddrvab.c",   # DSGESV mixed-precision solver
    "test_ddrvac.c",   # DSPOSV mixed-precision solver
    "test_dsgesv.c",   # DSGESV smoke test (mixed-precision)
    "dpot06.c",        # Mixed-precision residual verification
    "dget08.c",        # Mixed-precision residual verification
}

# Identifiers with embedded precision marker (not a simple d -> s prefix)
EMBEDDED_PRECISION = {
    "iladlc": "ilaslc",
    "iladlr": "ilaslr",
}

# Explicit filename renames for files where the precision marker is embedded
FILENAME_RENAMES = {
    "iladlc.c": "ilaslc.c",
    "iladlr.c": "ilaslr.c",
}

# Testdata header file renames
TESTDATA_RENAMES = {
    "dsx_testdata.h": "ssx_testdata.h",
    "dvx_testdata.h": "svx_testdata.h",
}

# LAPACK path prefixes: "DGE", "DGB", etc. -> "SGE", "SGB", etc.
# These appear in string literals in dlatb4.c, dlatb5.c, dlatb9.c, dlarhs.c
# and in test drivers. We handle them via string substitution rules.
LAPACK_PATH_PREFIXES = [
    "DGE", "DGB", "DPO", "DPP", "DPS", "DPB", "DPT", "DSY", "DSP",
    "DSR", "DSK", "DGT", "DQR", "DLQ", "DQL", "DRQ", "DQP", "DTZ",
    "DLS", "DLU", "DBD", "DBB", "DEC", "DHS", "DST", "DSG", "DBA",
    "DBL", "DBK", "DGL", "DGK", "DTR", "DTP", "DTB", "DRF",
    "DCK", "DGS", "DSB",
]


# ---------------------------------------------------------------------------
# Extract function names from headers
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


def extract_verify_names(header_path):
    """Return set of all d-prefixed identifiers declared in verify.h."""
    names = set()
    with open(header_path) as f:
        for line in f:
            # Match function declarations: type dname(
            # Handles: void dget01(, f64 dget06(, int dgennd(, f64 dqpt01(
            m = re.search(r"(?:void|f64|int)\s+(d\w+)\s*\(", line)
            if m:
                names.add(m.group(1))
    # Also pick up the struct/type name from dvx_testdata.h
    names.add("dvx_precomputed_t")
    return names


def extract_testdata_names(testdata_dir):
    """Return set of d-prefixed static array names from testdata headers."""
    names = set()
    for fname in os.listdir(testdata_dir):
        if not fname.endswith("_testdata.h"):
            continue
        if not fname.startswith("d"):
            continue
        fpath = os.path.join(testdata_dir, fname)
        with open(fpath) as f:
            for line in f:
                # static const f64 dsx_A_0[1] = {
                m = re.search(r"static\s+const\s+(?:f64|int)\s+(d\w+)", line)
                if m:
                    names.add(m.group(1))
                # Guard macros: DSX_TESTDATA_H, DVX_TESTDATA_H
                m = re.search(r"#(?:ifndef|define)\s+(D\w+_TESTDATA_H)", line)
                if m:
                    names.add(m.group(1))
    return names


# ---------------------------------------------------------------------------
# Build substitution rule lists
# ---------------------------------------------------------------------------
def build_code_subs(all_d_names):
    """Ordered (pattern, replacement) pairs for CODE segments."""
    subs = []

    # --- 1. Header include ---
    subs.append((re.compile(r"semicolon_lapack_double\.h"),
                 "semicolon_lapack_single.h"))

    # --- 2. CBLAS special: cblas_idamax -> cblas_isamax ---
    subs.append((re.compile(r"\bcblas_idamax\b"), "cblas_isamax"))

    # --- 3. CBLAS general: cblas_d* -> cblas_s* ---
    subs.append((re.compile(r"\bcblas_d(\w+)\b"), r"cblas_s\1"))

    # --- 4. ILA embedded precision ---
    for old, new in EMBEDDED_PRECISION.items():
        subs.append((re.compile(r"\b" + re.escape(old) + r"\b"), new))

    # --- 5. float.h constants ---
    subs.append((re.compile(r"\bDBL_EPSILON\b"), "FLT_EPSILON"))
    subs.append((re.compile(r"\bDBL_MIN_EXP\b"), "FLT_MIN_EXP"))
    subs.append((re.compile(r"\bDBL_MAX_EXP\b"), "FLT_MAX_EXP"))
    subs.append((re.compile(r"\bDBL_MANT_DIG\b"), "FLT_MANT_DIG"))
    subs.append((re.compile(r"\bDBL_MIN\b"), "FLT_MIN"))
    subs.append((re.compile(r"\bDBL_MAX\b"), "FLT_MAX"))

    # --- 5b. Type aliases ---
    subs.append((re.compile(r"\bf64\b"), "f32"))
    subs.append((re.compile(r"\bc128\b"), "c64"))

    # --- 5c. Guard macros (uppercase, before general name loop) ---
    subs.append((re.compile(r"\bDSX_TESTDATA_H\b"), "SSX_TESTDATA_H"))
    subs.append((re.compile(r"\bDVX_TESTDATA_H\b"), "SVX_TESTDATA_H"))
    subs.append((re.compile(r"\bVERIFY_H\b"), "VERIFY_H"))  # unchanged

    # --- 5d. RNG functions (no d-prefix, but precision-dependent) ---
    # Order: longest first to avoid partial matches
    subs.append((re.compile(r"\brng_uniform_symmetric\b"), "rng_uniform_symmetric_f32"))
    subs.append((re.compile(r"\brng_uniform\b"), "rng_uniform_f32"))
    subs.append((re.compile(r"\brng_normal\b"), "rng_normal_f32"))
    subs.append((re.compile(r"\brng_dist\b"), "rng_dist_f32"))
    subs.append((re.compile(r"\brng_fill\b"), "rng_fill_f32"))

    # --- 6. All d-prefixed names (longest first to avoid partial matches) ---
    for name in sorted(all_d_names, key=len, reverse=True):
        sname = "s" + name[1:]
        subs.append((re.compile(r"\b" + re.escape(name) + r"\b"), sname))

    # --- 7. Type keyword ---
    subs.append((re.compile(r"\bdouble\b"), "float"))

    # --- 8. Math functions ---
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
    subs.append((re.compile(r"(\d+\.\d+(?:[eE][+-]?\d+)?)(?!f\b)"),
                 r"\1f"))

    return subs


def build_string_subs(all_d_names):
    """Substitutions for string literals (uppercase xerbla names + paths)."""
    subs = []

    # Header include
    subs.append((re.compile(r"semicolon_lapack_double\.h"),
                 "semicolon_lapack_single.h"))

    # LAPACK path prefixes: "DGE" -> "SGE" etc.
    for prefix in LAPACK_PATH_PREFIXES:
        subs.append((re.compile(re.escape(prefix)), "S" + prefix[1:]))

    # Embedded precision identifiers (before general names)
    for old, new in EMBEDDED_PRECISION.items():
        subs.append((re.compile(re.escape(old.upper())), new.upper()))

    # Uppercase routine names: "DGETRF" -> "SGETRF"
    for name in sorted(all_d_names, key=len, reverse=True):
        upper_d = name.upper()
        upper_s = ("S" + name[1:]).upper()
        if upper_d != upper_s:
            subs.append((re.compile(re.escape(upper_d)), upper_s))

    # Testdata struct/array prefixes in strings (unlikely but safe)
    subs.append((re.compile(r"\bdsx_"), "ssx_"))
    subs.append((re.compile(r"\bdvx_"), "svx_"))

    return subs


def build_line_subs():
    """Substitutions applied to full lines BEFORE tokenization.

    These handle patterns that span code/char-literal boundaries,
    which the tokenizer would otherwise split apart.
    """
    subs = []
    # Precision marker in path character comparisons:
    #   path[0] == 'D' -> path[0] == 'S'
    #   path[0] == 'd' -> path[0] == 's'
    # This does NOT affect dlaord.c which uses job[0], not path[0].
    subs.append((re.compile(r"(path\[0\]\s*==\s*)'D'"), r"\1'S'"))
    subs.append((re.compile(r"(path\[0\]\s*==\s*)'d'"), r"\1's'"))
    subs.append((re.compile(r"(path\[0\]\s*!=\s*)'D'"), r"\1'S'"))
    subs.append((re.compile(r"(path\[0\]\s*!=\s*)'d'"), r"\1's'"))
    return subs


def build_comment_subs(all_d_names):
    """Substitutions for comment segments (@file tags, routine names)."""
    subs = []

    # @file renames: test_d*.c -> test_s*.c first, then d*.c -> s*.c
    subs.append((re.compile(r"(@file\s+test_)d(\w+\.c)"), r"\1s\2"))
    subs.append((re.compile(r"(@file\s+)d(\w+\.c)"), r"\1s\2"))
    subs.append((re.compile(r"(@file\s+)d(\w+\.h)"), r"\1s\2"))
    for old_f, new_f in FILENAME_RENAMES.items():
        subs.append((re.compile(r"(@file\s+)" + re.escape(old_f)),
                     r"\1" + new_f))
    for old_f, new_f in TESTDATA_RENAMES.items():
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

    # LAPACK path prefixes in comments
    for prefix in LAPACK_PATH_PREFIXES:
        subs.append((re.compile(r"\b" + re.escape(prefix) + r"\b"),
                     "S" + prefix[1:]))

    # Routine names in both cases: DGETRF -> SGETRF, dgetrf -> sgetrf
    for name in sorted(all_d_names, key=len, reverse=True):
        sname = "s" + name[1:]
        # uppercase
        subs.append((re.compile(r"\b" + re.escape(name.upper()) + r"\b"),
                     sname.upper()))
        # lowercase
        subs.append((re.compile(r"\b" + re.escape(name) + r"\b"), sname))

    # Testdata prefixes
    subs.append((re.compile(r"\bDSX_TESTDATA_H\b"), "SSX_TESTDATA_H"))
    subs.append((re.compile(r"\bDVX_TESTDATA_H\b"), "SVX_TESTDATA_H"))
    subs.append((re.compile(r"\bdsx_"), "ssx_"))
    subs.append((re.compile(r"\bdvx_"), "svx_"))
    subs.append((re.compile(r"\bDGEESX\b"), "SGEESX"))
    subs.append((re.compile(r"\bDGEEVX\b"), "SGEEVX"))

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
# Process a single C/H file
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


def process_file(src_path, code_subs, string_subs, comment_subs, line_subs):
    """Read a double-precision file and return single-precision content."""
    with open(src_path) as f:
        lines = f.readlines()

    output = []
    in_block = False

    for line in lines:
        # Apply line-level subs before tokenization (handles cross-boundary patterns)
        if not in_block:
            line = apply_subs(line, line_subs)
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


# ---------------------------------------------------------------------------
# Filename handling
# ---------------------------------------------------------------------------
def rename_file(name):
    """dXXX.c -> sXXX.c, with special cases."""
    if name in FILENAME_RENAMES:
        return FILENAME_RENAMES[name]
    if name in TESTDATA_RENAMES:
        return TESTDATA_RENAMES[name]
    # test_dchkge.c -> test_schkge.c
    if name.startswith("test_d"):
        return "test_s" + name[6:]
    if name.startswith("d"):
        return "s" + name[1:]
    return name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate single-precision LAPACK test files from double"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print converted output instead of writing files"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Process only this file (basename, e.g. dget01.c or test_dchkge.c)"
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Extract all d-prefixed routine names
    # -----------------------------------------------------------------------
    print(f"Reading LAPACK names from {DOUBLE_HEADER} ...", file=sys.stderr)
    lapack_names = extract_lapack_names(DOUBLE_HEADER)
    print(f"  Found {len(lapack_names)} d-prefixed library identifiers",
          file=sys.stderr)

    print(f"Reading verify names from {VERIFY_HEADER} ...", file=sys.stderr)
    verify_names = extract_verify_names(VERIFY_HEADER)
    print(f"  Found {len(verify_names)} d-prefixed test utility identifiers",
          file=sys.stderr)

    print(f"Reading testdata names from {TESTUTILS_D} ...", file=sys.stderr)
    testdata_names = extract_testdata_names(TESTUTILS_D)
    print(f"  Found {len(testdata_names)} d-prefixed testdata identifiers",
          file=sys.stderr)

    # Merge all names, but exclude uppercase guard macros (handled separately)
    guard_macros = {n for n in testdata_names if n.isupper()}
    all_d_names = lapack_names | verify_names | (testdata_names - guard_macros)

    # Add error exit routine
    all_d_names.add("derrec")

    print(f"  Total: {len(all_d_names)} unique d-prefixed identifiers",
          file=sys.stderr)

    # -----------------------------------------------------------------------
    # Build substitution tables
    # -----------------------------------------------------------------------
    code_subs = build_code_subs(all_d_names)
    string_subs = build_string_subs(all_d_names)
    comment_subs = build_comment_subs(all_d_names)
    line_subs = build_line_subs()

    # -----------------------------------------------------------------------
    # Determine file sets
    # -----------------------------------------------------------------------
    if args.file:
        # Locate the file in the tree
        candidates = []
        for root_dir in [TESTUTILS_D, TESTS_D, SMOKE_D]:
            fpath = os.path.join(root_dir, args.file)
            if os.path.exists(fpath):
                candidates.append((root_dir, args.file))
        if not candidates:
            print(f"ERROR: {args.file} not found in tests/d/ tree",
                  file=sys.stderr)
            sys.exit(1)
        file_sets = candidates
    else:
        file_sets = []
        # Testutils .c files
        for f in sorted(os.listdir(TESTUTILS_D)):
            if f.endswith(".c"):
                file_sets.append((TESTUTILS_D, f))
        # Testutils .h files (testdata headers)
        for f in sorted(os.listdir(TESTUTILS_D)):
            if f.endswith("_testdata.h") and f.startswith("d"):
                file_sets.append((TESTUTILS_D, f))
        # verify.h
        file_sets.append((TESTUTILS_D, "verify.h"))
        # Test driver files
        for f in sorted(os.listdir(TESTS_D)):
            if f.endswith(".c") and f.startswith("test_d"):
                file_sets.append((TESTS_D, f))
        # Smoke test files
        for f in sorted(os.listdir(SMOKE_D)):
            if f.endswith(".c") and f.startswith("test_d"):
                file_sets.append((SMOKE_D, f))

    # -----------------------------------------------------------------------
    # Clean output directories (full run only)
    # -----------------------------------------------------------------------
    if not args.dry_run and not args.file:
        for d in [TESTUTILS_S, SMOKE_S, TESTS_S]:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith(".c") or f.endswith(".h"):
                        os.remove(os.path.join(d, f))

    # -----------------------------------------------------------------------
    # Process C/H files
    # -----------------------------------------------------------------------
    converted = 0
    skipped = 0

    for src_dir, fname in file_sets:
        if fname in EXCLUDE_FILES:
            print(f"  SKIP (mixed-precision): {fname}", file=sys.stderr)
            skipped += 1
            continue

        src_path = os.path.join(src_dir, fname)
        dst_name = rename_file(fname)

        # Determine output directory
        if src_dir == TESTUTILS_D:
            dst_dir = TESTUTILS_S
        elif src_dir == SMOKE_D:
            dst_dir = SMOKE_S
        else:
            dst_dir = TESTS_S

        dst_path = os.path.join(dst_dir, dst_name)

        content = process_file(src_path, code_subs, string_subs, comment_subs,
                               line_subs)

        if args.dry_run:
            print(f"=== {fname} -> {dst_name} ===", file=sys.stderr)
            print(content)
        else:
            os.makedirs(dst_dir, exist_ok=True)
            with open(dst_path, "w") as f:
                f.write(content)
            converted += 1

    if not args.dry_run:
        print(f"\nConverted {converted} files, skipped {skipped}",
              file=sys.stderr)
        print(f"Output directories:", file=sys.stderr)
        print(f"  {TESTS_S}/", file=sys.stderr)
        print(f"  {TESTUTILS_S}/", file=sys.stderr)
        print(f"  {SMOKE_S}/", file=sys.stderr)


if __name__ == "__main__":
    main()
