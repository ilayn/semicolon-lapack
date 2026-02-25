#!/usr/bin/env python3
"""
Generate Fortran-ABI compatibility shim for semicolon-lapack.

Parses the C precision headers and generates wrapper functions that:
1. Add trailing underscore to symbol names (Fortran name mangling)
2. Convert pass-by-value scalars to pass-by-pointer (Fortran ABI)
3. Apply 0-based to 1-based index conversions (pivot arrays, ilo/ihi, etc.)

Usage:
    python scripts/generate_fortran_shim.py --input-dir src/include --output-dir src/fortran_shim
"""

import re
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Function pointer typedefs — these are pointer types despite having no '*'
# ---------------------------------------------------------------------------
FUNCPTR_TYPES = frozenset({
    'dselect2_t', 'dselect3_t',
    'sselect2_t', 'sselect3_t',
    'zselect1_t', 'zselect2_t',
    'cselect1_t', 'cselect2_t',
})

# ---------------------------------------------------------------------------
# Index conversion rules
#
# Keyed by base name (function name with single-char precision prefix stripped).
# Each value is a list of conversion actions:
#
#   ('out_pivot',      param, size_expr)           — output pivot array: ++ after call
#   ('in_pivot',       param, size_expr)           — input pivot array: -- copy before call
#   ('io_pivot',       param, size_expr, cond_param, cond_val)
#                                                  — conditional I/O: branch on char param
#   ('out_scalar_idx', param)                      — output scalar index: ++ after call
#   ('in_scalar_idx',  param)                      — input scalar index: -- before call
#   ('io_scalar_idx',  param)                      — I/O scalar index: -- before, ++ after
#   ('out_idx_arr',    param, size_expr)           — output index array: ++ after call
#   ('cond_out_idx_arr', param, size_expr, cond_param, cond_expr)
#                                                  — conditional output index array: ++ only
#                                                    when cond_expr is true (C expression using
#                                                    the char* parameter cond_param)
#   ('in_idx_arr',     param, size_expr)           — input index array: -- copy before call
#   ('gebal_scale',    param, ilo_param, ihi_param, n_param)
#                                                  — gebal: scale[0..ilo-2] and scale[ihi..n-1]
#                                                    contain permutation indices (0-based in C),
#                                                    convert to 1-based after ilo/ihi are incremented
#   ('gebak_scale',    param, ilo_param, ihi_param, n_param, job_param)
#                                                  — gebak: reverse of gebal_scale.
#                                                    Before call: convert scale perm indices from
#                                                    1-based to 0-based (using original 1-based ilo/ihi).
#                                                    After call: restore to 1-based.
#                                                    Only when job is 'P' or 'B'.
#
# Size expressions use shim parameter names with * dereference (e.g. '*n').
# They are evaluated in C at the point where they appear in the generated code.
# ---------------------------------------------------------------------------
INDEX_RULES = {
    # --- LU factorization family ---
    'getrf':  [('out_pivot', 'ipiv', '((*m < *n) ? *m : *n)')],
    'getf2':  [('out_pivot', 'ipiv', '((*m < *n) ? *m : *n)')],
    'getrf2': [('out_pivot', 'ipiv', '((*m < *n) ? *m : *n)')],
    'getrs':  [('in_pivot',  'ipiv', '*n')],
    'getri':  [('in_pivot',  'ipiv', '*n')],
    'gesv':   [('out_pivot', 'ipiv', '*n')],
    'getc2':  [('out_pivot', 'ipiv', '*n'), ('out_pivot', 'jpiv', '*n')],
    'gesc2':  [('in_pivot',  'ipiv', '*n'), ('in_pivot',  'jpiv', '*n')],
    'gerfs':  [('in_pivot',  'ipiv', '*n')],
    'gesvx':  [('io_pivot',  'ipiv', '*n', 'fact', 'F')],

    # --- Banded LU ---
    'gbtrf':  [('out_pivot', 'ipiv', '((*m < *n) ? *m : *n)')],
    'gbtf2':  [('out_pivot', 'ipiv', '((*m < *n) ? *m : *n)')],
    'gbtrs':  [('in_pivot',  'ipiv', '*n')],
    'gbsv':   [('out_pivot', 'ipiv', '*n')],
    'gbsvx':  [('io_pivot',  'ipiv', '*n', 'fact', 'F')],
    'gbrfs':  [('in_pivot',  'ipiv', '*n')],
    'gbcon':  [('in_pivot',  'ipiv', '*n')],

    # --- Tridiagonal ---
    'gttrf':  [('out_pivot', 'ipiv', '*n')],
    'gttrs':  [('in_pivot',  'ipiv', '*n')],
    'gtts2':  [('in_pivot',  'ipiv', '*n')],
    'gtsvx':  [('io_pivot',  'ipiv', '*n', 'fact', 'F')],
    'gtrfs':  [('in_pivot',  'ipiv', '*n')],
    'gtcon':  [('in_pivot',  'ipiv', '*n')],

    # --- Symmetric factorization ---
    'sytrf':      [('out_pivot', 'ipiv', '*n')],
    'sytf2':      [('out_pivot', 'ipiv', '*n')],
    'sytrs':      [('in_pivot',  'ipiv', '*n')],
    'sytrs2':     [('in_pivot',  'ipiv', '*n')],
    'sytri':      [('in_pivot',  'ipiv', '*n')],
    'sytri2':     [('in_pivot',  'ipiv', '*n')],
    'sytri2x':    [('in_pivot',  'ipiv', '*n')],
    'sycon':      [('in_pivot',  'ipiv', '*n')],
    'sysv':       [('out_pivot', 'ipiv', '*n')],
    'sysvx':      [('io_pivot',  'ipiv', '*n', 'fact', 'F')],
    'syrfs':      [('in_pivot',  'ipiv', '*n')],
    'syconv':     [('in_pivot',  'ipiv', '*n')],
    'syconvf':    [('in_pivot',  'ipiv', '*n')],
    'syconvf_rook': [('in_pivot', 'ipiv', '*n')],

    # --- Symmetric RK (bounded Bunch-Kaufman) ---
    'sytrf_rk':   [('out_pivot', 'ipiv', '*n')],
    'sytrs_3':    [('in_pivot',  'ipiv', '*n')],
    'sytri_3':    [('in_pivot',  'ipiv', '*n')],
    'sytri_3x':   [('in_pivot',  'ipiv', '*n')],
    'sycon_3':    [('in_pivot',  'ipiv', '*n')],
    'sysv_rk':    [('out_pivot', 'ipiv', '*n')],

    # --- Symmetric Rook ---
    'sytrf_rook': [('out_pivot', 'ipiv', '*n')],
    'sytrs_rook': [('in_pivot',  'ipiv', '*n')],
    'sytri_rook': [('in_pivot',  'ipiv', '*n')],
    'sycon_rook': [('in_pivot',  'ipiv', '*n')],
    'sysv_rook':  [('out_pivot', 'ipiv', '*n')],

    # --- Symmetric Aasen ---
    'sytrf_aa':   [('out_pivot', 'ipiv', '*n')],
    'sytrs_aa':   [('in_pivot',  'ipiv', '*n')],
    'sysv_aa':    [('out_pivot', 'ipiv', '*n')],

    # --- Symmetric Aasen 2-stage ---
    'sytrf_aa_2stage': [('out_pivot', 'ipiv', '*n'), ('out_pivot', 'ipiv2', '*n')],
    'sytrs_aa_2stage': [('in_pivot',  'ipiv', '*n'), ('in_pivot',  'ipiv2', '*n')],
    'sysv_aa_2stage':  [('out_pivot', 'ipiv', '*n'), ('out_pivot', 'ipiv2', '*n')],

    # --- Hermitian (complex only) ---
    'hetrf':      [('out_pivot', 'ipiv', '*n')],
    'hetf2':      [('out_pivot', 'ipiv', '*n')],
    'hetrs':      [('in_pivot',  'ipiv', '*n')],
    'hetrs2':     [('in_pivot',  'ipiv', '*n')],
    'hetri':      [('in_pivot',  'ipiv', '*n')],
    'hetri2':     [('in_pivot',  'ipiv', '*n')],
    'hetri2x':    [('in_pivot',  'ipiv', '*n')],
    'hecon':      [('in_pivot',  'ipiv', '*n')],
    'hesv':       [('out_pivot', 'ipiv', '*n')],
    'hesvx':      [('io_pivot',  'ipiv', '*n', 'fact', 'F')],
    'herfs':      [('in_pivot',  'ipiv', '*n')],

    'hetrf_rk':   [('out_pivot', 'ipiv', '*n')],
    'hetrs_3':    [('in_pivot',  'ipiv', '*n')],
    'hetri_3':    [('in_pivot',  'ipiv', '*n')],
    'hetri_3x':   [('in_pivot',  'ipiv', '*n')],
    'hecon_3':    [('in_pivot',  'ipiv', '*n')],
    'hesv_rk':    [('out_pivot', 'ipiv', '*n')],

    'hetrf_rook': [('out_pivot', 'ipiv', '*n')],
    'hetrs_rook': [('in_pivot',  'ipiv', '*n')],
    'hetri_rook': [('in_pivot',  'ipiv', '*n')],
    'hecon_rook': [('in_pivot',  'ipiv', '*n')],
    'hesv_rook':  [('out_pivot', 'ipiv', '*n')],

    'hetrf_aa':   [('out_pivot', 'ipiv', '*n')],
    'hetrs_aa':   [('in_pivot',  'ipiv', '*n')],
    'hesv_aa':    [('out_pivot', 'ipiv', '*n')],

    'hetrf_aa_2stage': [('out_pivot', 'ipiv', '*n'), ('out_pivot', 'ipiv2', '*n')],
    'hetrs_aa_2stage': [('in_pivot',  'ipiv', '*n'), ('in_pivot',  'ipiv2', '*n')],
    'hesv_aa_2stage':  [('out_pivot', 'ipiv', '*n'), ('out_pivot', 'ipiv2', '*n')],

    # --- Packed symmetric/Hermitian ---
    'sptrf':  [('out_pivot', 'ipiv', '*n')],
    'sptrs':  [('in_pivot',  'ipiv', '*n')],
    'sptri':  [('in_pivot',  'ipiv', '*n')],
    'spcon':  [('in_pivot',  'ipiv', '*n')],
    'spsv':   [('out_pivot', 'ipiv', '*n')],
    'spsvx':  [('io_pivot',  'ipiv', '*n', 'fact', 'F')],
    'sprfs':  [('in_pivot',  'ipiv', '*n')],

    'hptrf':  [('out_pivot', 'ipiv', '*n')],
    'hptrs':  [('in_pivot',  'ipiv', '*n')],
    'hptri':  [('in_pivot',  'ipiv', '*n')],
    'hpcon':  [('in_pivot',  'ipiv', '*n')],
    'hpsv':   [('out_pivot', 'ipiv', '*n')],
    'hpsvx':  [('io_pivot',  'ipiv', '*n', 'fact', 'F')],
    'hprfs':  [('in_pivot',  'ipiv', '*n')],

    # Panel factorizations (lasyf*, lahef*) are internal routines called by
    # sytrf/hetrf. They have inconsistent parameter naming (m vs n) and are
    # never called directly by SciPy. Omitted — simple pass-through wrappers.

    # --- Mixed precision (base name after stripping first char) ---
    'sgesv':  [('out_pivot', 'ipiv', '*n')],   # dsgesv → strip 'd' → 'sgesv'
    'cgesv':  [('out_pivot', 'ipiv', '*n')],   # zcgesv → strip 'z' → 'cgesv'

    # --- Cholesky pivoted ---
    'pstrf':  [('out_pivot', 'piv', '*n')],

    # --- QR with column pivoting ---
    'geqp3':   [('out_pivot', 'jpvt', '*n')],
    'geqp3rk': [('out_pivot', 'jpiv', '*n')],
    'gelsy':   [('in_pivot',  'jpvt', '*n')],
    'laqp2':   [('out_pivot', 'jpvt', '*n')],
    'laqps':   [('out_pivot', 'jpvt', '*n')],

    # --- latdf ---
    'latdf':  [('in_pivot', 'ipiv', '*n')],

    # --- Row interchange ---
    'laswp':  [('in_scalar_idx', 'k1'), ('in_scalar_idx', 'k2'),
               ('in_pivot', 'ipiv', '*k2')],

    # --- Balance / Hessenberg / eigenvalue ---
    'gebal':  [('out_scalar_idx', 'ilo'), ('out_scalar_idx', 'ihi'),
               ('gebal_scale', 'scale', 'ilo', 'ihi', 'n')],
    'gebak':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi'),
               ('gebak_scale', 'scale', 'ilo', 'ihi', 'n', 'job')],
    'ggbal':  [('out_scalar_idx', 'ilo'), ('out_scalar_idx', 'ihi')],
    'ggbak':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],
    'gehrd':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],
    'gehd2':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],
    'orghr':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],
    'unghr':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],
    'ormhr':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],
    'unmhr':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],
    'hseqr':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],
    'gghrd':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],
    'gghd3':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],
    'hgeqz':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],
    'laqr0':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],
    'laqr4':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],
    'laqz0':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],
    'laqz3':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],
    'laqz4':  [('in_scalar_idx',  'ilo'), ('in_scalar_idx',  'ihi')],

    'geevx':  [('out_scalar_idx', 'ilo'), ('out_scalar_idx', 'ihi')],
    'ggevx':  [('out_scalar_idx', 'ilo'), ('out_scalar_idx', 'ihi')],

    # --- Schur reordering (real: ifst/ilst are I/O pointers) ---
    'trexc':  [('io_scalar_idx', 'ifst'), ('io_scalar_idx', 'ilst')],
    'tgexc':  [('io_scalar_idx', 'ifst'), ('io_scalar_idx', 'ilst')],

    # Complex trexc/tgexc have different signatures: scalars by value, not pointers.
    # These full-name entries override the base-name rules above.
    'ztrexc': [('in_scalar_idx', 'ifst'), ('in_scalar_idx', 'ilst')],
    'ctrexc': [('in_scalar_idx', 'ifst'), ('in_scalar_idx', 'ilst')],
    'ztgexc': [('in_scalar_idx', 'ifst'), ('io_scalar_idx', 'ilst')],
    'ctgexc': [('in_scalar_idx', 'ifst'), ('io_scalar_idx', 'ilst')],

    # --- Eigenvalue range (il, iu) ---
    'stebz':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'iblock', '*m'),
               ('out_idx_arr', 'isplit', '*nsplit')],
    'stein':  [('in_idx_arr', 'iblock', '*m'), ('in_idx_arr', 'isplit', '*n'),
               ('out_idx_arr', 'ifail', '*m')],
    'stevx':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'ifail', '*m')],
    'stevr':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'isuppz', '2 * (*m)')],
    'stegr':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'isuppz', '2 * (*m)')],
    'stemr':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'isuppz', '2 * (*m)')],
    'syevx':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'ifail', '*m')],
    'syevr':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('cond_out_idx_arr', 'isuppz', '2 * (*m)', 'range', "(*jobz == 'V' || *jobz == 'v') && (*range == 'A' || *range == 'a')")],
    'syevx_2stage': [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
                     ('out_idx_arr', 'ifail', '*m')],
    'syevr_2stage': [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
                     ('cond_out_idx_arr', 'isuppz', '2 * (*m)', 'range', "(*jobz == 'V' || *jobz == 'v') && (*range == 'A' || *range == 'a')")],
    'heevx':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'ifail', '*m')],
    'heevr':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('cond_out_idx_arr', 'isuppz', '2 * (*m)', 'range', "(*jobz == 'V' || *jobz == 'v') && (*range == 'A' || *range == 'a')")],
    'heevx_2stage': [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
                     ('out_idx_arr', 'ifail', '*m')],
    'heevr_2stage': [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
                     ('cond_out_idx_arr', 'isuppz', '2 * (*m)', 'range', "(*jobz == 'V' || *jobz == 'v') && (*range == 'A' || *range == 'a')")],
    'spevx':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'ifail', '*m')],
    'hpevx':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'ifail', '*m')],
    'sbevx':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'ifail', '*m')],
    'hbevx':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'ifail', '*m')],
    'sbevx_2stage': [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
                     ('out_idx_arr', 'ifail', '*m')],
    'hbevx_2stage': [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
                     ('out_idx_arr', 'ifail', '*m')],
    'sygvx':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'ifail', '*m')],
    'sbgvx':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'ifail', '*m')],
    'hbgvx':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'ifail', '*m')],
    'spgvx':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'ifail', '*m')],
    'hpgvx':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'ifail', '*m')],

    'gesvdx': [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu')],
    'bdsvdx': [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu')],

    # --- Internal eigenvalue routines ---
    'larrd':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'iblock', '*m'),
               ('out_idx_arr', 'isplit', '*nsplit'),
               ('out_idx_arr', 'indexw', '*m')],
    'larre':  [('in_scalar_idx', 'il'), ('in_scalar_idx', 'iu'),
               ('out_idx_arr', 'iblock', '*m'),
               ('out_idx_arr', 'isplit', '*nsplit'),
               ('out_idx_arr', 'indexw', '*m')],
    'larrv':  [('in_idx_arr', 'isplit', '*n'),
               ('in_idx_arr', 'iblock', '*m'),
               ('in_idx_arr', 'indexw', '*m'),
               ('out_idx_arr', 'isuppz', '2 * (*m)')],

    # --- Eigenvector failure arrays ---
    'hsein':  [('out_idx_arr', 'ifaill', '*mm'),
               ('out_idx_arr', 'ifailr', '*mm')],

    # --- Secular equation eigenvalue index ---
    'lasd4':  [('in_scalar_idx', 'i')],
    'lasd5':  [('in_scalar_idx', 'i')],

    # --- Cholesky pivoted (unblocked) ---
    'pstf2':  [('out_pivot', 'piv', '*n')],
}


# ---------------------------------------------------------------------------
# Header parser
# ---------------------------------------------------------------------------

# Match: SEMICOLON_API <return_type> <name>(<params>);
DECL_RE = re.compile(
    r'SEMICOLON_API\s+'
    r'(\S+(?:\s+\S+)?)\s+'    # return type (e.g. "void", "f64", "c128")
    r'(\w+)'                   # function name
    r'\(([^)]*)\)\s*;'         # parameter list
)

def _parse_param(param_str):
    """Parse a single parameter into (type, name).

    Handles both 'INT* info' and 'INT *info' styles.
    """
    param_str = param_str.strip()
    # The parameter name is the last word (sequence of [a-zA-Z0-9_])
    m = re.search(r'(\w+)\s*$', param_str)
    if not m:
        return (param_str, '')
    name = m.group(1)
    type_str = param_str[:m.start()].strip()
    # Normalize: collapse 'INT *' or 'INT * restrict' to 'INT*' / 'INT* restrict'
    type_str = re.sub(r'\s+\*', '*', type_str)
    return (type_str, name)


def parse_declarations(header_path):
    """Parse SEMICOLON_API declarations from a header file.

    Returns list of (return_type, func_name, [(param_type, param_name), ...])
    """
    text = Path(header_path).read_text()
    results = []

    for m in DECL_RE.finditer(text):
        ret_type = m.group(1).strip()
        func_name = m.group(2).strip()
        param_str = m.group(3).strip()

        if not param_str:
            results.append((ret_type, func_name, []))
            continue

        params = []
        for p in param_str.split(','):
            params.append(_parse_param(p))

        results.append((ret_type, func_name, params))

    return results


# ---------------------------------------------------------------------------
# Parameter classification
# ---------------------------------------------------------------------------

def is_scalar(param_type):
    """True if the parameter is a scalar (no pointer in type, not a funcptr typedef)."""
    if '*' in param_type:
        return False
    # Strip const and check against known function pointer typedefs
    base = param_type.replace('const', '').strip()
    if base in FUNCPTR_TYPES:
        return False
    return True


def make_shim_type(param_type, scalar):
    """Convert our API parameter type to the shim's Fortran-ABI type."""
    # Strip const and restrict
    t = param_type
    t = re.sub(r'\bconst\b', '', t)
    t = re.sub(r'\brestrict\b', '', t)
    t = ' '.join(t.split())  # normalize whitespace
    if not t:
        t = 'void'
    if scalar:
        t = t + '*'
    return t


def get_base_name(func_name, prefix_char):
    """Strip single-char precision prefix to get base name for rule lookup."""
    if func_name and func_name[0] == prefix_char and len(func_name) > 1:
        return func_name[1:]
    return func_name


# ---------------------------------------------------------------------------
# Wrapper code generation
# ---------------------------------------------------------------------------

def generate_wrapper(func_name, return_type, params, rules):
    """Generate the complete shim wrapper function as a list of C lines."""
    lines = []

    # Build parameter info
    param_info = []
    for ptype, pname in params:
        sc = is_scalar(ptype)
        shim_type = make_shim_type(ptype, sc)
        param_info.append({
            'orig_type': ptype,
            'name': pname,
            'scalar': sc,
            'shim_type': shim_type,
        })

    # Build rule lookup by parameter name
    rule_map = {}  # param_name -> list of rules
    for r in rules:
        pname = r[1]
        rule_map.setdefault(pname, []).append(r)

    # Check for io_pivot rules (need special branching)
    io_pivot_rules = [r for r in rules if r[0] == 'io_pivot']

    # Detect workspace query parameters (lwork, liwork, lrwork)
    lwork_params = [p['name'] for p in param_info
                    if p['name'] in ('lwork', 'liwork', 'lrwork')]

    # --- Shim signature ---
    shim_params_str = ', '.join(f'{p["shim_type"]} {p["name"]}' for p in param_info)
    ret_clean = return_type.replace('const', '').replace('restrict', '').strip()
    ret_clean = ' '.join(ret_clean.split())

    # --- io_pivot branch case ---
    if io_pivot_rules:
        return _generate_io_pivot_wrapper(
            func_name, ret_clean, param_info, io_pivot_rules, rules, shim_params_str)

    # --- Standard wrapper ---
    lines.append(f'{ret_clean} {func_name}_({shim_params_str}) {{')

    # Determine which rules need workspace-query guarding.
    # When lwork==-1 the routine only computes optimal workspace sizes; pivot
    # and index arrays are untouched.  Callers (e.g. f2py) may pass scalar
    # placeholders for those arrays, so iterating over them would corrupt the
    # stack.  We guard array-level conversions with `if (*lwork != -1)`.
    needs_lwork_guard = bool(lwork_params) and any(
        r[0] in ('out_pivot', 'in_pivot', 'in_idx_arr', 'out_idx_arr',
                 'cond_out_idx_arr')
        for r in rules)

    # Pre-call: in_scalar_idx locals, in_pivot/in_idx_arr temp copies
    pre_lines = []
    free_list = []

    # Scalar index conversions are always safe (they operate on single scalars)
    for r in rules:
        action = r[0]
        pname = r[1]
        if action == 'in_scalar_idx':
            pre_lines.append(f'    INT _{pname} = *{pname} - 1;')
        elif action == 'io_scalar_idx':
            pre_lines.append(f'    INT _{pname} = *{pname} - 1;')
        elif action == 'gebak_scale':
            # Convert scale permutation indices from 1-based to 0-based
            # before calling our library.  Uses original (1-based) ilo/ihi.
            ilo_p = r[2]
            ihi_p = r[3]
            n_p = r[4]
            job_p = r[5]
            pre_lines.append(
                f'    if ({job_p}[0] == \'B\' || {job_p}[0] == \'b\' || '
                f'{job_p}[0] == \'P\' || {job_p}[0] == \'p\') {{')
            pre_lines.append(
                f'        for (INT _i = 0; _i < *{ilo_p} - 1; _i++) {pname}[_i] -= 1.0;')
            pre_lines.append(
                f'        for (INT _i = *{ihi_p}; _i < *{n_p}; _i++) {pname}[_i] -= 1.0;')
            pre_lines.append(f'    }}')

    # Array conversions: guarded if routine has lwork
    arr_decl_lines = []   # variable declarations (always emitted, for scope)
    arr_init_lines = []   # allocation + copy (guarded when lwork present)
    for r in rules:
        action = r[0]
        pname = r[1]
        if action in ('in_pivot', 'in_idx_arr'):
            size_expr = r[2]
            tmp = f'_{pname}0'
            arr_decl_lines.append(f'    INT* {tmp} = NULL;')
            arr_init_lines.append(f'    if ({pname}) {{')
            arr_init_lines.append(f'        INT _{pname}_sz = {size_expr};')
            arr_init_lines.append(f'        {tmp} = (INT*)malloc(_{pname}_sz * sizeof(INT));')
            arr_init_lines.append(f'        for (INT _i = 0; _i < _{pname}_sz; _i++) {tmp}[_i] = ({pname}[_i] > 0) ? {pname}[_i] - 1 : {pname}[_i];')
            arr_init_lines.append(f'    }}')
            free_list.append(tmp)

    lines.extend(pre_lines)

    # Declarations are always emitted (needed for call args and free)
    lines.extend(arr_decl_lines)

    if needs_lwork_guard and arr_init_lines:
        lines.append(f'    if (*{lwork_params[0]} != -1) {{')
        for al in arr_init_lines:
            lines.append('    ' + al)
        lines.append('    }')
    else:
        lines.extend(arr_init_lines)

    # Function call — when lwork guard is active and there are in_pivot params,
    # we need two call paths: one with temp copies, one without
    call_args = []
    call_args_passthru = []  # for lwork==-1 path: pass original arrays
    for p in param_info:
        pname = p['name']
        prules = rule_map.get(pname, [])

        if any(r[0] == 'in_scalar_idx' for r in prules):
            call_args.append(f'_{pname}')
            call_args_passthru.append(f'_{pname}')
        elif any(r[0] == 'io_scalar_idx' for r in prules):
            call_args.append(f'&_{pname}')
            call_args_passthru.append(f'&_{pname}')
        elif any(r[0] in ('in_pivot', 'in_idx_arr') for r in prules):
            call_args.append(f'_{pname}0')
            call_args_passthru.append(pname)
        elif p['scalar']:
            call_args.append(f'*{pname}')
            call_args_passthru.append(f'*{pname}')
        else:
            call_args.append(pname)
            call_args_passthru.append(pname)

    has_in_arr_rules = any(r[0] in ('in_pivot', 'in_idx_arr') for r in rules)

    if needs_lwork_guard and has_in_arr_rules:
        # Two call paths: guarded uses temp copies, query path passes originals
        call_str = ', '.join(call_args)
        call_str_passthru = ', '.join(call_args_passthru)
        if ret_clean == 'void':
            lines.append(f'    if (*{lwork_params[0]} != -1) {{')
            lines.append(f'        {func_name}({call_str});')
            lines.append(f'    }} else {{')
            lines.append(f'        {func_name}({call_str_passthru});')
            lines.append(f'    }}')
        else:
            lines.append(f'    {ret_clean} _ret;')
            lines.append(f'    if (*{lwork_params[0]} != -1) {{')
            lines.append(f'        _ret = {func_name}({call_str});')
            lines.append(f'    }} else {{')
            lines.append(f'        _ret = {func_name}({call_str_passthru});')
            lines.append(f'    }}')
    else:
        call_str = ', '.join(call_args)
        if ret_clean == 'void':
            lines.append(f'    {func_name}({call_str});')
        else:
            lines.append(f'    {ret_clean} _ret = {func_name}({call_str});')

    # Post-call: out_pivot, out_scalar_idx, out_idx_arr, io_scalar_idx writeback
    post_arr_lines = []
    post_scalar_lines = []
    for r in rules:
        action = r[0]
        pname = r[1]

        if action == 'out_pivot':
            size_expr = r[2]
            post_arr_lines.append(f'    if ({pname}) {{ INT _sz = {size_expr}; for (INT _i = 0; _i < _sz; _i++) {{ if ({pname}[_i] >= 0) {pname}[_i]++; }} }}')

        elif action == 'out_scalar_idx':
            post_scalar_lines.append(f'    (*{pname})++;')

        elif action == 'out_idx_arr':
            size_expr = r[2]
            post_arr_lines.append(f'    if ({pname}) {{ INT _sz = {size_expr}; for (INT _i = 0; _i < _sz; _i++) {pname}[_i]++; }}')

        elif action == 'cond_out_idx_arr':
            size_expr = r[2]
            cond_expr = r[4]
            post_arr_lines.append(f'    if ({pname} && ({cond_expr})) {{ INT _sz = {size_expr}; for (INT _i = 0; _i < _sz; _i++) {pname}[_i]++; }}')

        elif action == 'io_scalar_idx':
            post_scalar_lines.append(f'    *{pname} = _{pname} + 1;')

        elif action == 'gebal_scale':
            # scale[0..ilo-2] and scale[ihi..n-1] contain 0-based permutation
            # indices that must be converted to 1-based.  By this point, ilo
            # and ihi have already been incremented to 1-based Fortran values.
            ilo_p = r[2]
            ihi_p = r[3]
            n_p = r[4]
            post_scalar_lines.append(
                f'    for (INT _i = 0; _i < *{ilo_p} - 1; _i++) {pname}[_i] += 1.0;')
            post_scalar_lines.append(
                f'    for (INT _i = *{ihi_p}; _i < *{n_p}; _i++) {pname}[_i] += 1.0;')

        elif action == 'gebak_scale':
            # Restore scale permutation indices to 1-based after call.
            # Uses the original (1-based) ilo/ihi from the caller.
            ilo_p = r[2]
            ihi_p = r[3]
            n_p = r[4]
            job_p = r[5]
            post_scalar_lines.append(
                f'    if ({job_p}[0] == \'B\' || {job_p}[0] == \'b\' || '
                f'{job_p}[0] == \'P\' || {job_p}[0] == \'p\') {{')
            post_scalar_lines.append(
                f'        for (INT _i = 0; _i < *{ilo_p} - 1; _i++) {pname}[_i] += 1.0;')
            post_scalar_lines.append(
                f'        for (INT _i = *{ihi_p}; _i < *{n_p}; _i++) {pname}[_i] += 1.0;')
            post_scalar_lines.append(f'    }}')

    # Scalar post-call conversions are always safe
    lines.extend(post_scalar_lines)

    # Array post-call conversions: guarded if routine has lwork
    if needs_lwork_guard and post_arr_lines:
        lines.append(f'    if (*{lwork_params[0]} != -1) {{')
        for pl in post_arr_lines:
            lines.append('    ' + pl)
        lines.append('    }')
    else:
        lines.extend(post_arr_lines)

    # Free temp copies (also guarded — they weren't allocated during query)
    if needs_lwork_guard and free_list:
        lines.append(f'    if (*{lwork_params[0]} != -1) {{')
        for tmp in free_list:
            lines.append(f'        free({tmp});')
        lines.append('    }')
    else:
        for tmp in free_list:
            lines.append(f'    free({tmp});')

    # Return
    if ret_clean != 'void':
        lines.append('    return _ret;')

    lines.append('}')
    return lines


def _generate_io_pivot_wrapper(func_name, ret_clean, param_info, io_rules,
                               all_rules, shim_params_str):
    """Generate wrapper for expert drivers with conditional I/O pivots."""
    lines = []
    lines.append(f'{ret_clean} {func_name}_({shim_params_str}) {{')

    # All io_pivot rules share the same condition parameter
    cond_param = io_rules[0][3]
    cond_val = io_rules[0][4]

    lines.append(f'    if ({cond_param}[0] == \'{cond_val}\' || '
                 f'{cond_param}[0] == \'{cond_val.lower()}\') {{')

    # INPUT branch: decrement copies, call, free
    free_list = []
    for r in io_rules:
        pname = r[1]
        size_expr = r[2]
        tmp = f'_{pname}0'
        lines.append(f'        INT* {tmp} = NULL;')
        lines.append(f'        if ({pname}) {{')
        lines.append(f'            INT _{pname}_sz = {size_expr};')
        lines.append(f'            {tmp} = (INT*)malloc(_{pname}_sz * sizeof(INT));')
        lines.append(f'            for (INT _i = 0; _i < _{pname}_sz; _i++) {tmp}[_i] = ({pname}[_i] > 0) ? {pname}[_i] - 1 : {pname}[_i];')
        lines.append(f'        }}')
        free_list.append(tmp)

    # Build call args for input branch
    rule_map = {}
    for r in io_rules:
        rule_map.setdefault(r[1], []).append(r)

    call_args = []
    for p in param_info:
        pname = p['name']
        if pname in rule_map:
            call_args.append(f'_{pname}0')
        elif p['scalar']:
            call_args.append(f'*{pname}')
        else:
            call_args.append(pname)

    call_str = ', '.join(call_args)
    if ret_clean == 'void':
        lines.append(f'        {func_name}({call_str});')
    else:
        lines.append(f'        {ret_clean} _ret = {func_name}({call_str});')

    for tmp in free_list:
        lines.append(f'        free({tmp});')

    if ret_clean != 'void':
        lines.append('        return _ret;')

    lines.append('    } else {')

    # OUTPUT branch: call, then increment
    call_args2 = []
    for p in param_info:
        if p['scalar']:
            call_args2.append(f'*{p["name"]}')
        else:
            call_args2.append(p['name'])

    call_str2 = ', '.join(call_args2)
    if ret_clean == 'void':
        lines.append(f'        {func_name}({call_str2});')
    else:
        lines.append(f'        {ret_clean} _ret = {func_name}({call_str2});')

    for r in io_rules:
        pname = r[1]
        size_expr = r[2]
        lines.append(f'        if ({pname}) {{ INT _sz = {size_expr}; for (INT _i = 0; _i < _sz; _i++) {{ if ({pname}[_i] >= 0) {pname}[_i]++; }} }}')

    if ret_clean != 'void':
        lines.append('        return _ret;')

    lines.append('    }')
    lines.append('}')
    return lines


# ---------------------------------------------------------------------------
# File generation
# ---------------------------------------------------------------------------

HEADER_MAP = {
    'double':         ('semicolon_lapack_double.h', 'd'),
    'single':         ('semicolon_lapack_single.h', 's'),
    'complex_double': ('semicolon_lapack_complex_double.h', 'z'),
    'complex_single': ('semicolon_lapack_complex_single.h', 'c'),
    'auxiliary':      ('semicolon_lapack_auxiliary.h', ''),
}

# Functions to skip (static inline helpers, not real API)
SKIP_FUNCTIONS = frozenset({
    'cabs1', 'cabs2', 'cabs1f', 'cabs2f',
})


def generate_shim_file(input_dir, precision, output_path):
    """Parse a header and generate the complete shim C file."""
    header_name, prefix_char = HEADER_MAP[precision]
    header_path = Path(input_dir) / header_name
    decls = parse_declarations(header_path)

    lines = []
    lines.append('/* AUTO-GENERATED FILE — DO NOT EDIT')
    lines.append(' * Fortran-ABI compatibility shim for semicolon-lapack')
    lines.append(f' * Precision: {precision}')
    lines.append(f' * Source header: {header_name}')
    lines.append(' * Generated by scripts/generate_fortran_shim.py')
    lines.append(' */')
    lines.append('')
    lines.append('#include <stdlib.h>')

    # Include the precision header
    if precision == 'auxiliary':
        lines.append('#include "semicolon_lapack_auxiliary.h"')
    else:
        lines.append(f'#include "{header_name}"')
    lines.append('')

    n_simple = 0
    n_indexed = 0
    n_skipped = 0

    for ret_type, func_name, params in decls:
        if func_name in SKIP_FUNCTIONS:
            n_skipped += 1
            continue

        # Look up index conversion rules (full name first, then base name)
        rules = INDEX_RULES.get(func_name, None)
        if rules is None:
            base = get_base_name(func_name, prefix_char) if prefix_char else func_name
            rules = INDEX_RULES.get(base, [])

        wrapper_lines = generate_wrapper(func_name, ret_type, params, rules)
        lines.append('')
        lines.extend(wrapper_lines)

        if rules:
            n_indexed += 1
        else:
            n_simple += 1

    # Write output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n')

    return n_simple, n_indexed, n_skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate Fortran-ABI compatibility shim for semicolon-lapack')
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing source headers (src/include)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for generated shim files (src/fortran_shim)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        'double':         'shim_double.c',
        'single':         'shim_single.c',
        'complex_double': 'shim_complex_double.c',
        'complex_single': 'shim_complex_single.c',
        'auxiliary':      'shim_auxiliary.c',
    }

    total_simple = 0
    total_indexed = 0
    total_skipped = 0

    for precision, filename in file_map.items():
        out_path = output_dir / filename
        n_s, n_i, n_sk = generate_shim_file(args.input_dir, precision, out_path)
        total_simple += n_s
        total_indexed += n_i
        total_skipped += n_sk
        print(f'{filename}: {n_s} simple + {n_i} indexed + {n_sk} skipped = {n_s + n_i + n_sk} total')

    print(f'\nTotal: {total_simple} simple + {total_indexed} indexed + {total_skipped} skipped '
          f'= {total_simple + total_indexed + total_skipped} functions')


if __name__ == '__main__':
    main()
