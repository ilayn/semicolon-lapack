#!/usr/bin/env python3
"""Generate complex single precision (c64) test files from complex double (c128).

Usage:
    python scripts/gen_complex_single_tests.py              # convert all files
    python scripts/gen_complex_single_tests.py --dry-run    # preview without writing
    python scripts/gen_complex_single_tests.py --file zget01.c           # single file (testutils)
    python scripts/gen_complex_single_tests.py --file test_zchkge.c      # single file (driver)
    python scripts/gen_complex_single_tests.py --file test_zchkge.c --dry-run

Reads tests/z/ tree, applies token-aware precision substitutions, writes tests/c/ tree.
Comments are left unchanged except for @file tags and uppercase routine names.
String literals only get uppercase routine name substitutions (for xerbla) and
precision-path substitutions ("ZGE" -> "CGE").
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
TESTS_Z = os.path.join(PROJECT_ROOT, "tests", "z")
TESTS_C = os.path.join(PROJECT_ROOT, "tests", "c")
TESTUTILS_Z = os.path.join(TESTS_Z, "testutils")
TESTUTILS_C = os.path.join(TESTS_C, "testutils")
COMPLEX_DOUBLE_HEADER = os.path.join(PROJECT_ROOT, "src", "include",
                                     "semicolon_lapack_complex_double.h")
DOUBLE_HEADER = os.path.join(PROJECT_ROOT, "src", "include",
                             "semicolon_lapack_double.h")
VERIFY_HEADER = os.path.join(TESTUTILS_Z, "verify.h")

# Files to skip (mixed-precision tests with no c-prefix analogue)
EXCLUDE_FILES = {
    "test_zdrvab.c",   # ZCGESV mixed-precision solver
    "test_zdrvac.c",   # ZCPOSV mixed-precision solver
}

# Identifiers with embedded precision marker (not a simple z -> c prefix)
EMBEDDED_PRECISION = {
    "ilazlc": "ilaclc",
    "ilazlr": "ilaclr",
}

# Explicit filename renames for files where the precision marker is embedded
FILENAME_RENAMES = {
    "ilazlc.c": "ilaclc.c",
    "ilazlr.c": "ilaclr.c",
}

# Cross-precision prefix renames (z-files that have non-z prefixes)
CROSS_PREFIX_FILENAMES = {
    "dzsum1.c": "scsum1.c",
    "izmax1.c": "icmax1.c",
}

# Testdata header file renames
TESTDATA_RENAMES = {
    "zsx_testdata.h": "csx_testdata.h",
    "zvx_testdata.h": "cvx_testdata.h",
    "zgsx_testdata.h": "cgsx_testdata.h",
    "zgvx_testdata.h": "cgvx_testdata.h",
}

# LAPACK path prefixes: "ZGE", "ZPO", etc. -> "CGE", "CPO", etc.
# These appear in string literals in zlatb4.c, zlatb5.c, zlarhs.c
# and in test drivers.
LAPACK_PATH_PREFIXES = [
    "ZGE", "ZGB", "ZGT", "ZHE", "ZHP", "ZLQ", "ZPB", "ZPO", "ZPP",
    "ZPS", "ZPT", "ZQK", "ZQL", "ZQR", "ZRQ", "ZSP", "ZSY", "ZTB",
    "ZTP", "ZTR", "ZBD", "ZBB", "ZEC", "ZHS", "ZST", "ZSG", "ZBA",
    "ZBL", "ZBK", "ZGL", "ZGK", "ZRF", "ZCK", "ZGS", "ZSB", "ZLS",
]


# ---------------------------------------------------------------------------
# Extract function names from headers
# ---------------------------------------------------------------------------
def extract_z_lapack_names(header_path):
    """Return set of all z-prefixed (and dz/iz) identifiers declared in the header."""
    names = set()
    with open(header_path) as f:
        for line in f:
            m = re.search(r"SEMICOLON_API\s+\w[\w*\s]*\s+(z\w+|dz\w+|iz\w+)\s*\(", line)
            if m:
                names.add(m.group(1))
    # callback typedefs
    names.add("zselect1_t")
    names.add("zselect2_t")
    return names


def extract_d_lapack_names(header_path):
    """Return set of all d-prefixed identifiers declared in the double header."""
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
    """Return set of all z-prefixed identifiers declared in verify.h."""
    names = set()
    with open(header_path) as f:
        for line in f:
            # Match function declarations: type zname( or type dzname(
            m = re.search(r"(?:void|f64|INT|int|c128)\s+(z\w+|dz\w+|iz\w+)\s*\(", line)
            if m:
                names.add(m.group(1))
            # Also pick up d-prefixed shared routines
            m = re.search(r"(?:void|f64|INT|int)\s+(d\w+)\s*\(", line)
            if m:
                names.add(m.group(1))
    return names


def extract_testdata_names(testdata_dir):
    """Return set of z-prefixed static array names from testdata headers."""
    names = set()
    for fname in os.listdir(testdata_dir):
        if not fname.endswith("_testdata.h"):
            continue
        if not fname.startswith("z"):
            continue
        fpath = os.path.join(testdata_dir, fname)
        with open(fpath) as f:
            for line in f:
                # static const c128 zsx_A_0[1] = {
                # static const f64 zgvx_dtru_0[4] = {
                # static const INT zsx_islct_0[1] = {
                m = re.search(r"static\s+const\s+(?:c128|f64|INT|int)\s+(z\w+)", line)
                if m:
                    names.add(m.group(1))
                # Guard macros: ZSX_TESTDATA_H, ZGSX_TESTDATA_H
                m = re.search(r"#(?:ifndef|define)\s+(Z\w+_TESTDATA_H)", line)
                if m:
                    names.add(m.group(1))
    return names


def z_to_c_name(name):
    """Convert a z-prefixed name to its c-prefixed equivalent."""
    if name in EMBEDDED_PRECISION:
        return EMBEDDED_PRECISION[name]
    if name.startswith("dz"):
        return "sc" + name[2:]
    if name.startswith("iz"):
        return "ic" + name[2:]
    if name.startswith("z"):
        return "c" + name[1:]
    return name


# ---------------------------------------------------------------------------
# Build substitution rule lists
# ---------------------------------------------------------------------------
def build_code_subs(z_names, d_names):
    """Ordered (pattern, replacement) pairs for CODE segments."""
    subs = []

    # --- 1. Header include ---
    subs.append((re.compile(r"semicolon_lapack_complex_double\.h"),
                 "semicolon_lapack_complex_single.h"))

    # --- 2. CBLAS special mixed-precision: cblas_zdscal -> cblas_csscal ---
    subs.append((re.compile(r"\bcblas_zdscal\b"), "cblas_csscal"))
    subs.append((re.compile(r"\bcblas_zdrot\b"), "cblas_csrot"))

    # --- 3. CBLAS real-from-complex: cblas_dz* -> cblas_sc* ---
    subs.append((re.compile(r"\bcblas_dznrm2\b"), "cblas_scnrm2"))
    subs.append((re.compile(r"\bcblas_dzasum\b"), "cblas_scasum"))

    # --- 4. CBLAS special: cblas_izamax -> cblas_icamax ---
    subs.append((re.compile(r"\bcblas_izamax\b"), "cblas_icamax"))

    # --- 5. CBLAS dot product subs ---
    subs.append((re.compile(r"\bcblas_zdotc_sub\b"), "cblas_cdotc_sub"))
    subs.append((re.compile(r"\bcblas_zdotu_sub\b"), "cblas_cdotu_sub"))

    # --- 6. CBLAS general: cblas_z* -> cblas_c* ---
    subs.append((re.compile(r"\bcblas_z(\w+)\b"), r"cblas_c\1"))

    # --- 6b. CBLAS pure-real calls: cblas_idamax -> cblas_isamax, cblas_d* -> cblas_s* ---
    subs.append((re.compile(r"\bcblas_idamax\b"), "cblas_isamax"))
    subs.append((re.compile(r"\bcblas_d(\w+)\b"), r"cblas_s\1"))

    # --- 7. Complex C functions ---
    subs.append((re.compile(r"\bCMPLX\b"), "CMPLXF"))
    subs.append((re.compile(r"\bcreal\b"), "crealf"))
    subs.append((re.compile(r"\bcimag\b"), "cimagf"))
    subs.append((re.compile(r"\bconj\b"), "conjf"))
    # Complex math: csqrt -> csqrtf, cexp -> cexpf, etc. (C11 <complex.h>)
    subs.append((re.compile(r"\bcsqrt\b"), "csqrtf"))
    subs.append((re.compile(r"\bcexp\b"), "cexpf"))
    subs.append((re.compile(r"\bclog\b"), "clogf"))
    subs.append((re.compile(r"\bcpow\b"), "cpowf"))
    subs.append((re.compile(r"\bcabs\b"), "cabsf"))

    # --- 8. Helper macros from header: cabs1 -> cabs1f, cabs2 -> cabs2f ---
    subs.append((re.compile(r"\bcabs1\b"), "cabs1f"))
    subs.append((re.compile(r"\bcabs2\b"), "cabs2f"))

    # --- 9. ILA embedded precision ---
    for old, new in EMBEDDED_PRECISION.items():
        subs.append((re.compile(r"\b" + re.escape(old) + r"\b"), new))

    # --- 10. float.h constants ---
    subs.append((re.compile(r"\bDBL_EPSILON\b"), "FLT_EPSILON"))
    subs.append((re.compile(r"\bDBL_MIN_EXP\b"), "FLT_MIN_EXP"))
    subs.append((re.compile(r"\bDBL_MAX_EXP\b"), "FLT_MAX_EXP"))
    subs.append((re.compile(r"\bDBL_MANT_DIG\b"), "FLT_MANT_DIG"))
    subs.append((re.compile(r"\bDBL_MIN\b"), "FLT_MIN"))
    subs.append((re.compile(r"\bDBL_MAX\b"), "FLT_MAX"))

    # --- 11. Type aliases ---
    subs.append((re.compile(r"\bc128\b"), "c64"))
    subs.append((re.compile(r"\bf64\b"), "f32"))

    # --- 11b. Guard macros (uppercase, before general name loop) ---
    subs.append((re.compile(r"\bZSX_TESTDATA_H\b"), "CSX_TESTDATA_H"))
    subs.append((re.compile(r"\bZVX_TESTDATA_H\b"), "CVX_TESTDATA_H"))
    subs.append((re.compile(r"\bZGSX_TESTDATA_H\b"), "CGSX_TESTDATA_H"))
    subs.append((re.compile(r"\bZGVX_TESTDATA_H\b"), "CGVX_TESTDATA_H"))
    subs.append((re.compile(r"\bVERIFY_Z_H\b"), "VERIFY_C_H"))

    # --- 11c. RNG functions (no z-prefix, but precision-dependent) ---
    subs.append((re.compile(r"\brng_uniform_symmetric\b"), "rng_uniform_symmetric_f32"))
    subs.append((re.compile(r"\brng_uniform\b"), "rng_uniform_f32"))
    subs.append((re.compile(r"\brng_normal\b"), "rng_normal_f32"))
    subs.append((re.compile(r"\brng_dist\b"), "rng_dist_f32"))
    subs.append((re.compile(r"\brng_fill\b"), "rng_fill_f32"))

    # --- 12. z-prefixed names (longest first) ---
    for name in sorted(z_names, key=len, reverse=True):
        cname = z_to_c_name(name)
        subs.append((re.compile(r"\b" + re.escape(name) + r"\b"), cname))

    # --- 13. d-prefixed names (real helpers, longest first) ---
    for name in sorted(d_names, key=len, reverse=True):
        sname = "s" + name[1:]
        subs.append((re.compile(r"\b" + re.escape(name) + r"\b"), sname))

    # --- 14. Math functions ---
    math_funcs = [
        ("copysign", "copysignf"),
        ("log10",    "log10f"),
        ("log2",     "log2f"),
        ("atan2",    "atan2f"),
        ("ldexp",    "ldexpf"),
        ("floor",    "floorf"),
        ("round",    "roundf"),
        ("sqrt",     "sqrtf"),
        ("fabs",     "fabsf"),
        ("fmax",     "fmaxf"),
        ("fmin",     "fminf"),
        ("fmod",     "fmodf"),
        ("ceil",     "ceilf"),
        ("pow",      "powf"),
        ("log",      "logf"),
        ("tan",      "tanf"),
        ("sin",      "sinf"),
        ("cos",      "cosf"),
    ]
    for old, new in math_funcs:
        subs.append((re.compile(r"\b" + re.escape(old) + r"\b"), new))

    # --- 15. Float literals: 1.0 -> 1.0f  (not already suffixed) ---
    subs.append((re.compile(r"(\d+\.\d+(?:[eE][+-]?\d+)?)(?!f\b)"),
                 r"\1f"))

    return subs


def build_string_subs(z_names, d_names):
    """Substitutions for string literals (uppercase xerbla names + paths)."""
    subs = []

    # Header include
    subs.append((re.compile(r"semicolon_lapack_complex_double\.h"),
                 "semicolon_lapack_complex_single.h"))

    # LAPACK path prefixes: "ZGE" -> "CGE" etc.
    for prefix in LAPACK_PATH_PREFIXES:
        subs.append((re.compile(re.escape(prefix)), "C" + prefix[1:]))

    # Embedded precision identifiers (before general names)
    for old, new in EMBEDDED_PRECISION.items():
        subs.append((re.compile(re.escape(old.upper())), new.upper()))

    # Uppercase z-routine names: "ZGETRF" -> "CGETRF"
    for name in sorted(z_names, key=len, reverse=True):
        upper_z = name.upper()
        upper_c = z_to_c_name(name).upper()
        if upper_z != upper_c:
            subs.append((re.compile(re.escape(upper_z)), upper_c))

    # Uppercase d-routine names: "DLAMCH" -> "SLAMCH"
    for name in sorted(d_names, key=len, reverse=True):
        upper_d = name.upper()
        upper_s = ("S" + name[1:]).upper()
        if upper_d != upper_s:
            subs.append((re.compile(re.escape(upper_d)), upper_s))

    # Testdata struct/array prefixes in strings
    subs.append((re.compile(r"\bzsx_"), "csx_"))
    subs.append((re.compile(r"\bzvx_"), "cvx_"))
    subs.append((re.compile(r"\bzgsx_"), "cgsx_"))
    subs.append((re.compile(r"\bzgvx_"), "cgvx_"))

    return subs


def build_line_subs():
    """Substitutions applied to full lines BEFORE tokenization.

    These handle patterns that span code/char-literal boundaries,
    which the tokenizer would otherwise split apart.
    """
    subs = []
    # Precision marker in path character comparisons:
    #   path[0] == 'Z' -> path[0] == 'C'
    #   path[0] == 'z' -> path[0] == 'c'
    # This does NOT affect pack[0] == 'Z' (where 'Z' means tridiagonal band format).
    subs.append((re.compile(r"(path\[0\]\s*==\s*)'Z'"), r"\1'C'"))
    subs.append((re.compile(r"(path\[0\]\s*==\s*)'z'"), r"\1'c'"))
    subs.append((re.compile(r"(path\[0\]\s*!=\s*)'Z'"), r"\1'C'"))
    subs.append((re.compile(r"(path\[0\]\s*!=\s*)'z'"), r"\1'c'"))
    return subs


def build_comment_subs(z_names, d_names):
    """Substitutions for comment segments (@file tags, routine names)."""
    subs = []

    # @file renames: test_z*.c -> test_c*.c first, then z*.c -> c*.c
    subs.append((re.compile(r"(@file\s+test_)z(\w+\.c)"), r"\1c\2"))
    subs.append((re.compile(r"(@file\s+)z(\w+\.c)"), r"\1c\2"))
    subs.append((re.compile(r"(@file\s+)z(\w+\.h)"), r"\1c\2"))
    for old_f, new_f in FILENAME_RENAMES.items():
        subs.append((re.compile(r"(@file\s+)" + re.escape(old_f)),
                     r"\1" + new_f))
    for old_f, new_f in CROSS_PREFIX_FILENAMES.items():
        subs.append((re.compile(r"(@file\s+)" + re.escape(old_f)),
                     r"\1" + new_f))
    for old_f, new_f in TESTDATA_RENAMES.items():
        subs.append((re.compile(r"(@file\s+)" + re.escape(old_f)),
                     r"\1" + new_f))

    # float.h constants in comments
    subs.append((re.compile(r"\bDBL_MIN_EXP\b"), "FLT_MIN_EXP"))
    subs.append((re.compile(r"\bDBL_MAX_EXP\b"), "FLT_MAX_EXP"))
    subs.append((re.compile(r"\bDBL_MANT_DIG\b"), "FLT_MANT_DIG"))

    # Type keywords in comments
    subs.append((re.compile(r"\bc128\b"), "c64"))
    subs.append((re.compile(r"\bf64\b"), "f32"))
    subs.append((re.compile(r"\bDouble complex\b"), "Single complex"))
    subs.append((re.compile(r"\bdouble complex\b"), "single complex"))
    subs.append((re.compile(r"\bcomplex double\b"), "complex single"))
    subs.append((re.compile(r"\bComplex double\b"), "Complex single"))
    subs.append((re.compile(r"\bz-prefix\b"), "c-prefix"))

    # Embedded precision identifiers (before general LAPACK names)
    for old, new in EMBEDDED_PRECISION.items():
        subs.append((re.compile(r"\b" + re.escape(old.upper()) + r"\b"),
                     new.upper()))
        subs.append((re.compile(r"\b" + re.escape(old) + r"\b"), new))

    # LAPACK path prefixes in comments
    for prefix in LAPACK_PATH_PREFIXES:
        subs.append((re.compile(r"\b" + re.escape(prefix) + r"\b"),
                     "C" + prefix[1:]))

    # z-routine names in both cases: ZGETRF -> CGETRF, zgetrf -> cgetrf
    for name in sorted(z_names, key=len, reverse=True):
        cname = z_to_c_name(name)
        # uppercase
        subs.append((re.compile(r"\b" + re.escape(name.upper()) + r"\b"),
                     cname.upper()))
        # lowercase
        subs.append((re.compile(r"\b" + re.escape(name) + r"\b"), cname))

    # d-routine names in both cases (real helpers referenced in comments)
    for name in sorted(d_names, key=len, reverse=True):
        sname = "s" + name[1:]
        subs.append((re.compile(r"\b" + re.escape(name.upper()) + r"\b"),
                     sname.upper()))
        subs.append((re.compile(r"\b" + re.escape(name) + r"\b"), sname))

    # Testdata prefixes
    subs.append((re.compile(r"\bZSX_TESTDATA_H\b"), "CSX_TESTDATA_H"))
    subs.append((re.compile(r"\bZVX_TESTDATA_H\b"), "CVX_TESTDATA_H"))
    subs.append((re.compile(r"\bZGSX_TESTDATA_H\b"), "CGSX_TESTDATA_H"))
    subs.append((re.compile(r"\bZGVX_TESTDATA_H\b"), "CGVX_TESTDATA_H"))
    subs.append((re.compile(r"\bzsx_"), "csx_"))
    subs.append((re.compile(r"\bzvx_"), "cvx_"))
    subs.append((re.compile(r"\bzgsx_"), "cgsx_"))
    subs.append((re.compile(r"\bzgvx_"), "cgvx_"))

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
    """Read a complex-double test file and return complex-single content."""
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
    """zXXX.c -> cXXX.c, with special cases."""
    if name in FILENAME_RENAMES:
        return FILENAME_RENAMES[name]
    if name in CROSS_PREFIX_FILENAMES:
        return CROSS_PREFIX_FILENAMES[name]
    if name in TESTDATA_RENAMES:
        return TESTDATA_RENAMES[name]
    # test_zchkge.c -> test_cchkge.c
    if name.startswith("test_z"):
        return "test_c" + name[6:]
    if name.startswith("z"):
        return "c" + name[1:]
    return name


# ---------------------------------------------------------------------------
# Meson.build transformation
# ---------------------------------------------------------------------------
def transform_meson_build(src_path, dst_path, is_testutils=False):
    """Transform a meson.build file with z->c replacements.

    Meson files are not C code, so we do simple text replacement
    rather than token-aware substitution.
    """
    with open(src_path) as f:
        text = f.read()

    # Library/dep names
    text = text.replace("testutils_z_sources", "testutils_c_sources")
    text = text.replace("'testutils_z'", "'testutils_c'")
    text = text.replace("libtestutils_z", "libtestutils_c")
    text = text.replace("verify_dep_z", "verify_dep_c")
    text = text.replace("test_inc_z", "test_inc_c")
    text = text.replace("lapack_tests_z", "lapack_tests_c")

    # Source file references: z*.c -> c*.c
    text = re.sub(r"'z(\w+\.c)'", r"'c\1'", text)
    text = re.sub(r"'test_z(\w+\.c)'", r"'test_c\1'", text)

    # Testdata headers
    for old_f, new_f in TESTDATA_RENAMES.items():
        text = text.replace(old_f, new_f)

    # Cross-prefix filenames
    for old_f, new_f in CROSS_PREFIX_FILENAMES.items():
        text = text.replace(old_f, new_f)

    # Test names (keys in the dict): 'zchkge' -> 'cchkge'
    text = re.sub(r"'z(chk\w+)'", r"'c\1'", text)
    text = re.sub(r"'z(drv\w+)'", r"'c\1'", text)
    text = re.sub(r"'z(ck\w+)'", r"'c\1'", text)
    text = re.sub(r"'z(dr\w+)'", r"'c\1'", text)

    # Shared real routines: d -> s
    text = text.replace("../../d/testutils/dget06.c", "../../s/testutils/sget06.c")
    text = text.replace("../../d/testutils/dsvdct.c", "../../s/testutils/ssvdct.c")
    text = text.replace("../../d/testutils/dsvdch.c", "../../s/testutils/ssvdch.c")
    text = text.replace("../../d/testutils/dstect.c", "../../s/testutils/sstect.c")
    text = text.replace("../../d/testutils/dstech.c", "../../s/testutils/sstech.c")
    text = text.replace("../../d/testutils/dsxt1.c", "../../s/testutils/ssxt1.c")
    text = text.replace("../../d/testutils/dlatm1.c", "../../s/testutils/slatm1.c")
    text = text.replace("../../d/testutils/dlatm7.c", "../../s/testutils/slatm7.c")
    text = text.replace("../../d/testutils/dlaord.c", "../../s/testutils/slaord.c")
    text = text.replace("../../d/testutils/dlatb9.c", "../../s/testutils/slatb9.c")

    # Comments
    text = text.replace("Complex double", "Complex single")
    text = text.replace("complex double", "complex single")
    text = text.replace("z-prefix", "c-prefix")
    text = text.replace("d-prefix", "s-prefix")

    # Comment references: zchk*.f -> cchk*.f, zdrv*.f -> cdrv*.f
    text = text.replace("zchk*.f", "cchk*.f")
    text = text.replace("zdrv*.f", "cdrv*.f")

    # Remove mixed-precision test entries
    text = re.sub(r"^\s*'cdrvab'.*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*'cdrvac'.*\n", "", text, flags=re.MULTILINE)

    with open(dst_path, "w") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate complex-single LAPACK test files from complex-double"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print converted output instead of writing files"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Process only this file (basename, e.g. zget01.c or test_zchkge.c)"
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Extract all z/d-prefixed routine names
    # -----------------------------------------------------------------------
    print(f"Reading z-LAPACK names from {COMPLEX_DOUBLE_HEADER} ...",
          file=sys.stderr)
    z_lapack_names = extract_z_lapack_names(COMPLEX_DOUBLE_HEADER)
    print(f"  Found {len(z_lapack_names)} z-prefixed library identifiers",
          file=sys.stderr)

    print(f"Reading d-LAPACK names from {DOUBLE_HEADER} ...", file=sys.stderr)
    d_lapack_names = extract_d_lapack_names(DOUBLE_HEADER)
    print(f"  Found {len(d_lapack_names)} d-prefixed library identifiers",
          file=sys.stderr)

    print(f"Reading verify names from {VERIFY_HEADER} ...", file=sys.stderr)
    verify_names = extract_verify_names(VERIFY_HEADER)
    z_verify = {n for n in verify_names if n.startswith("z") or n.startswith("dz") or n.startswith("iz")}
    d_verify = {n for n in verify_names if n.startswith("d") and not n.startswith("dz")}
    print(f"  Found {len(z_verify)} z-prefixed + {len(d_verify)} d-prefixed test utility identifiers",
          file=sys.stderr)

    print(f"Reading testdata names from {TESTUTILS_Z} ...", file=sys.stderr)
    testdata_names = extract_testdata_names(TESTUTILS_Z)
    z_testdata = {n for n in testdata_names if n[0].islower() and n.startswith("z")}
    z_testdata_guards = {n for n in testdata_names if n.isupper()}
    print(f"  Found {len(z_testdata)} z-prefixed testdata identifiers + {len(z_testdata_guards)} guard macros",
          file=sys.stderr)

    # Merge all z-prefixed names
    all_z_names = z_lapack_names | z_verify | z_testdata
    # Add error exit routine
    all_z_names.add("zerrec")

    # Merge all d-prefixed names (real helpers)
    all_d_names = d_lapack_names | d_verify

    print(f"  Total: {len(all_z_names)} z-prefixed + {len(all_d_names)} d-prefixed identifiers",
          file=sys.stderr)

    # -----------------------------------------------------------------------
    # Build substitution tables
    # -----------------------------------------------------------------------
    code_subs = build_code_subs(all_z_names, all_d_names)
    string_subs = build_string_subs(all_z_names, all_d_names)
    comment_subs = build_comment_subs(all_z_names, all_d_names)
    line_subs = build_line_subs()

    # -----------------------------------------------------------------------
    # Determine file sets
    # -----------------------------------------------------------------------
    if args.file:
        candidates = []
        for root_dir in [TESTUTILS_Z, TESTS_Z]:
            fpath = os.path.join(root_dir, args.file)
            if os.path.exists(fpath):
                candidates.append((root_dir, args.file))
        if not candidates:
            print(f"ERROR: {args.file} not found in tests/z/ tree",
                  file=sys.stderr)
            sys.exit(1)
        file_sets = candidates
    else:
        file_sets = []
        # Testutils .c files
        for f in sorted(os.listdir(TESTUTILS_Z)):
            if f.endswith(".c"):
                file_sets.append((TESTUTILS_Z, f))
        # Testutils .h files (testdata headers)
        for f in sorted(os.listdir(TESTUTILS_Z)):
            if f.endswith("_testdata.h") and f.startswith("z"):
                file_sets.append((TESTUTILS_Z, f))
        # verify.h
        file_sets.append((TESTUTILS_Z, "verify.h"))
        # Test driver files
        for f in sorted(os.listdir(TESTS_Z)):
            if f.endswith(".c") and f.startswith("test_z"):
                file_sets.append((TESTS_Z, f))

    # -----------------------------------------------------------------------
    # Clean output directories (full run only)
    # -----------------------------------------------------------------------
    if not args.dry_run and not args.file:
        for d in [TESTUTILS_C, TESTS_C]:
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
        if src_dir == TESTUTILS_Z:
            dst_dir = TESTUTILS_C
        else:
            dst_dir = TESTS_C

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

    # -----------------------------------------------------------------------
    # Process meson.build files (full run only)
    # -----------------------------------------------------------------------
    if not args.dry_run and not args.file:
        # testutils/meson.build
        src_meson = os.path.join(TESTUTILS_Z, "meson.build")
        dst_meson = os.path.join(TESTUTILS_C, "meson.build")
        os.makedirs(TESTUTILS_C, exist_ok=True)
        transform_meson_build(src_meson, dst_meson, is_testutils=True)
        print(f"  meson.build -> testutils/meson.build", file=sys.stderr)

        # tests/c/meson.build
        src_meson = os.path.join(TESTS_Z, "meson.build")
        dst_meson = os.path.join(TESTS_C, "meson.build")
        transform_meson_build(src_meson, dst_meson)
        print(f"  meson.build -> tests/c/meson.build", file=sys.stderr)

    if not args.dry_run:
        print(f"\nConverted {converted} files, skipped {skipped}",
              file=sys.stderr)
        print(f"Output directories:", file=sys.stderr)
        print(f"  {TESTS_C}/", file=sys.stderr)
        print(f"  {TESTUTILS_C}/", file=sys.stderr)


if __name__ == "__main__":
    main()
