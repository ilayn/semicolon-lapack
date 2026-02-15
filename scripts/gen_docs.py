#!/usr/bin/env python3
"""Generate the complete API documentation structure with precision tabs.

Creates leaf RST pages (with sphinx-design tab-set blocks where multiple
precisions exist), intermediate toctree pages, and the api/index.rst.
Idempotent: running twice produces the same output.

Usage:
    python scripts/gen_docs.py              # regenerate all RST files
    python scripts/gen_docs.py --dry-run    # preview without writing
"""

import os
import sys
import argparse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
API_DIR = os.path.join(PROJECT_ROOT, "doc", "source", "api")
SRC_D = os.path.join(PROJECT_ROOT, "src", "d")
SRC_S = os.path.join(PROJECT_ROOT, "src", "s")
SRC_C = os.path.join(PROJECT_ROOT, "src", "c")
SRC_Z = os.path.join(PROJECT_ROOT, "src", "z")
SRC_AUX = os.path.join(PROJECT_ROOT, "src", "auxiliary")

# ---------------------------------------------------------------------------
# Precision configuration (tab order: single first)
# ---------------------------------------------------------------------------
PRECISIONS = [
    ("s", "Single",         "single",         SRC_S),
    ("d", "Double",         "double",         SRC_D),
    ("c", "Complex Single", "complex-single", SRC_C),
    ("z", "Complex Double", "complex-double", SRC_Z),
]

# ---------------------------------------------------------------------------
# Source file mapping exceptions
# ---------------------------------------------------------------------------
# For standard pages: source = <prefix><page_name>.c (e.g., page "sysv" ->
# dsysv.c, ssysv.c, csysv.c, zsysv.c). Only pages that deviate from this
# pattern need entries here.
#
# Format: page_name -> {precision_prefix: [source_basenames_without_.c]}
# Precisions not listed are derived from the 'd' entry by swapping the first
# character, unless NO 'd' entry exists.
SPECIAL_SOURCES = {
    # Mixed precision (single source, never gets tabs)
    "gesv_mixed":  {"d": ["dsgesv"]},
    "posv_mixed":  {"d": ["dsposv"]},
    # Precision conversion routines
    "_lag2_":      {"d": ["dlag2s"]},
    "_lat2_":      {"d": ["dlat2s"]},
    "lag2d":       {"s": ["slag2d"]},
    # Integer functions with embedded precision character
    "ilalc":       {"d": ["iladlc"], "s": ["ilaslc"],
                    "c": ["ilaclc"], "z": ["ilazlc"]},
    "ilalr":       {"d": ["iladlr"], "s": ["ilaslr"],
                    "c": ["ilaclr"], "z": ["ilazlr"]},
    # Multi-file pages
    "larf":        {"d": ["dlarf", "dlarf1f", "dlarf1l"],
                    "s": ["slarf", "slarf1f", "slarf1l"],
                    "c": ["clarf", "clarf1f", "clarf1l"],
                    "z": ["zlarf", "zlarf1f", "zlarf1l"]},
    # Complex-only routines with non-standard naming
    "sum1":        {"z": ["dzsum1"], "c": ["scsum1"]},
    "max1":        {"z": ["izmax1"], "c": ["icmax1"]},
    "lag2c":       {"z": ["zlag2c"]},
    "lat2c":       {"z": ["zlat2c"]},
}

# Pages with no precision prefix (precision-independent utilities).
# Shown as bare doxygenfile directives, no tabs.
PRECISION_INDEPENDENT = {"xerbla", "ieeeck", "ilaenv2stage", "iparam2stage", "iparmq"}

# ---------------------------------------------------------------------------
# Hierarchy definition
# ---------------------------------------------------------------------------
# Structure: {dir_name: (title, {subdir: ...} | [page_names])}
# Page names use the LAPACK base name.  Real-only families (sy, sp, sb,
# or, op) and complex-only families (he, hp, hb, un, up) each get their
# own pages.  The script checks the filesystem to determine which
# precision tabs to show.
HIERARCHY = {
    "linear-systems": ("Linear Systems", {
        "general": ("General Matrix", [
            "gesv", "gesv_mixed", "gesvx",
            "getrf", "getrf2", "getf2", "getrs", "getri",
            "gecon", "geequ", "geequb", "gerfs",
            "getc2", "gesc2",
            "laswp", "laqge", "latdf",
        ]),
        "general-banded": ("General Banded", [
            "gbsv", "gbsvx",
            "gbtrf", "gbtf2", "gbtrs",
            "gbcon", "gbequ", "gbequb", "gbrfs",
            "laqgb",
        ]),
        "general-tridiagonal": ("General Tridiagonal", [
            "gtsv", "gtsvx",
            "gttrf", "gttrs", "gtts2",
            "gtcon", "gtrfs",
        ]),
        "spd": ("Symmetric Positive Definite", [
            "posv", "posv_mixed", "posvx",
            "potrf", "potrf2", "potf2", "potrs", "potri",
            "pocon", "poequ", "poequb", "porfs",
            "laqsy",
        ]),
        "spd-band": ("SPD Band Storage", [
            "pbsv", "pbsvx",
            "pbtrf", "pbtf2", "pbtrs",
            "pbcon", "pbequ", "pbrfs",
            "laqsb",
        ]),
        "spd-packed": ("SPD Packed Storage", [
            "ppsv", "ppsvx",
            "pptrf", "pptrs", "pptri",
            "ppcon", "ppequ", "pprfs",
            "laqsp",
        ]),
        "spd-rfp": ("SPD Rectangular Full Packed", [
            "pftrf", "pftrs", "pftri",
            "pstf2", "pstrf",
        ]),
        "spd-tridiagonal": ("SPD Tridiagonal", [
            "ptsv", "ptsvx",
            "pttrf", "pttrs", "ptts2",
            "ptcon", "ptrfs",
        ]),
        "symmetric-indefinite": ("Symmetric Indefinite", [
            "sysv", "sysv_rk", "sysv_rook",
            "sysv_aa", "sysv_aa_2stage",
            "sysvx",
            "sytrf", "sytrf_rk", "sytrf_rook",
            "sytrf_aa", "sytrf_aa_2stage",
            "sytf2", "sytf2_rk", "sytf2_rook",
            "sytrs", "sytrs2", "sytrs_3",
            "sytrs_rook",
            "sytrs_aa", "sytrs_aa_2stage",
            "sytri", "sytri2", "sytri2x",
            "sytri_3", "sytri_3x", "sytri_rook",
            "sycon", "sycon_3", "sycon_rook",
            "syequb", "syrfs", "syswapr",
            "lasyf", "lasyf_rk", "lasyf_rook", "lasyf_aa",
            "syconv", "syconvf", "syconvf_rook",
        ]),
        "symmetric-indefinite-packed": ("Symmetric Indefinite Packed", [
            "spsv", "spsvx",
            "sptrf", "sptrs", "sptri",
            "spcon", "sprfs",
        ]),
        "hermitian-indefinite": ("Hermitian Indefinite", [
            "hesv", "hesv_rk", "hesv_rook",
            "hesv_aa", "hesv_aa_2stage",
            "hesvx",
            "hetrf", "hetrf_rk", "hetrf_rook",
            "hetrf_aa", "hetrf_aa_2stage",
            "hetf2", "hetf2_rk", "hetf2_rook",
            "hetrs", "hetrs2", "hetrs_3",
            "hetrs_rook",
            "hetrs_aa", "hetrs_aa_2stage",
            "hetri", "hetri2", "hetri2x",
            "hetri_3", "hetri_3x", "hetri_rook",
            "hecon", "hecon_3", "hecon_rook",
            "heequb", "herfs", "heswapr",
            "lahef", "lahef_rk", "lahef_rook", "lahef_aa",
            "laqhe",
        ]),
        "hermitian-indefinite-packed": ("Hermitian Indefinite Packed", [
            "hpsv", "hpsvx",
            "hptrf", "hptrs", "hptri",
            "hpcon", "hprfs",
            "laqhp",
        ]),
        "triangular": ("Triangular", [
            "trtrs", "trtri", "trti2", "trcon", "trrfs",
            "latrs", "latrs3",
            "lauu2", "lauum",
        ]),
        "triangular-band": ("Triangular Band", [
            "tbtrs", "tbcon", "tbrfs",
            "latbs",
        ]),
        "triangular-packed": ("Triangular Packed", [
            "tptrs", "tptri", "tpcon", "tprfs",
            "latps",
        ]),
        "triangular-rfp": ("Triangular RFP", [
            "tftri",
        ]),
        "auxiliary": ("Condition Estimation Helpers", [
            "lacn2", "lacon",
            "la_gbamv", "la_gbrpvgrw",
            "la_geamv", "la_gerpvgrw",
            "la_lin_berr",
        ]),
    }),
    "least-squares": ("Least Squares", {
        "drivers": ("Drivers", [
            "gels", "gelsd", "gelss", "gelst", "gelsy", "getsls",
        ]),
        "constrained": ("Constrained", [
            "ggglm", "gglse",
        ]),
        "auxiliary": ("Auxiliary", [
            "laic1", "lals0", "lalsa", "lalsd",
        ]),
    }),
    "orthogonal": ("Orthogonal / Unitary Factors", {
        "qr": ("QR Factorization", [
            "geqrf", "geqr2", "geqr2p", "geqrfp",
            "geqr", "geqrt", "geqrt2", "geqrt3",
            "orgqr", "org2r", "ungqr", "ung2r",
            "ormqr", "orm2r", "unmqr", "unm2r",
            "gemqr", "gemqrt",
        ]),
        "qr-pivoting": ("QR with Column Pivoting", [
            "geqp3", "geqp3rk", "laqp2", "laqp2rk", "laqp3rk", "laqps",
        ]),
        "qr-tall-skinny": ("QR, Tall-Skinny (TSQR)", [
            "latsqr", "lamtsqr", "getsqrhrt", "larfb_gett",
            "orgtsqr", "orgtsqr_row", "ungtsqr", "ungtsqr_row",
            "laorhr_col_getrfnp", "laorhr_col_getrfnp2", "orhr_col",
            "launhr_col_getrfnp", "launhr_col_getrfnp2", "unhr_col",
        ]),
        "qr-tri-pent": ("QR, Triangular-Pentagonal", [
            "tpqrt", "tpqrt2", "tpmqrt", "tprfb",
        ]),
        "lq": ("LQ Factorization", [
            "gelqf", "gelq2", "gelq", "gelqt", "gelqt3",
            "orglq", "orgl2", "unglq", "ungl2",
            "ormlq", "orml2", "unmlq", "unml2",
            "gemlq", "gemlqt",
        ]),
        "lq-short-wide": ("LQ, Short-Wide", [
            "laswlq", "lamswlq",
        ]),
        "lq-tri-pent": ("LQ, Triangular-Pentagonal", [
            "tplqt", "tplqt2", "tpmlqt",
        ]),
        "ql": ("QL Factorization", [
            "geqlf", "geql2",
            "orgql", "org2l", "ungql", "ung2l",
            "ormql", "orm2l", "unmql", "unm2l",
        ]),
        "rq": ("RQ Factorization", [
            "gerqf", "gerq2",
            "orgrq", "orgr2", "ungrq", "ungr2",
            "ormrq", "ormr2", "unmrq", "unmr2",
        ]),
        "rz": ("RZ Factorization", [
            "tzrzf", "latrz",
            "ormrz", "ormr3", "unmrz", "unmr3",
            "larz", "larzb", "larzt",
        ]),
        "householder": ("Householder Reflectors", [
            "larfg", "larfgp",
            "larf", "larfb", "larft", "larfx", "larfy",
            "larft_lvl2",
        ]),
        "givens": ("Givens / Jacobi Rotations", [
            "lartg", "lar2v", "largv", "lartgp", "lartv", "lasr",
            "rot", "lacrt",
        ]),
        "cs-decomposition": ("Cosine-Sine (CS) Decomposition", [
            "orcsd", "orcsd2by1", "uncsd", "uncsd2by1",
            "bbcsd",
            "orbdb", "orbdb1", "orbdb2", "orbdb3",
            "orbdb4", "orbdb5", "orbdb6",
            "unbdb", "unbdb1", "unbdb2", "unbdb3",
            "unbdb4", "unbdb5", "unbdb6",
            "lapmr", "lapmt",
        ]),
        "generalized-qr": ("Generalized QR", [
            "ggqrf",
        ]),
        "generalized-rq": ("Generalized RQ", [
            "ggrqf",
        ]),
    }),
    "eigenvalues-nonsymmetric": ("Non-symmetric Eigenvalues", {
        "standard-drivers": ("Standard Eigenvalue Drivers", [
            "geev", "geevx", "gees", "geesx",
        ]),
        "generalized-drivers": ("Generalized Eigenvalue Drivers", [
            "ggev", "ggev3", "ggevx", "gges", "gges3", "ggesx",
        ]),
        "computational": ("Computational Routines", [
            "gehrd", "gehd2", "orghr", "ormhr", "unghr", "unmhr",
            "gebak", "gebal",
            "hseqr", "hsein",
            "trevc", "trevc3", "trexc", "trsen", "trsna",
            "trsyl", "trsyl3",
            "lahqr", "lahr2",
            "laqr0", "laqr1", "laqr2", "laqr3", "laqr4", "laqr5",
            "laqz0", "laqz1", "laqz2", "laqz3", "laqz4",
            "laein", "laexc", "laesy", "laln2", "lanv2", "laqtr", "lasy2",
            "iparmq",
        ]),
        "generalized-computational": ("Generalized Computational", [
            "gghrd", "gghd3", "ggbak", "ggbal",
            "hgeqz",
            "tgevc", "tgexc", "tgex2", "tgsen", "tgsna",
            "tgsy2", "tgsyl",
            "lagv2", "orm22", "unm22",
        ]),
    }),
    "eigenvalues-symmetric": ("Symmetric / Hermitian Eigenvalues", {
        "dense-drivers": ("Dense Matrix Drivers", [
            "syev", "syev_2stage",
            "syevd", "syevd_2stage",
            "syevr", "syevr_2stage",
            "syevx", "syevx_2stage",
        ]),
        "band-drivers": ("Band Matrix Drivers", [
            "sbev", "sbev_2stage",
            "sbevd", "sbevd_2stage",
            "sbevx", "sbevx_2stage",
        ]),
        "packed-drivers": ("Packed Matrix Drivers", [
            "spev", "spevd", "spevx",
        ]),
        "tridiagonal-drivers": ("Tridiagonal Eigensolvers", [
            "stev", "stevd", "stevr", "stevx",
            "steqr", "sterf", "stedc", "stemr", "stegr",
            "stebz", "stein",
            "pteqr",
        ]),
        "generalized-drivers": ("Generalized Eigenvalue Drivers", [
            "sygv", "sygv_2stage", "sygvd", "sygvx",
            "sbgv", "sbgvd", "sbgvx",
            "spgv", "spgvd", "spgvx",
        ]),
        "reduction": ("Reduction to Tridiagonal", [
            "sytrd", "sytrd_2stage", "sytrd_sy2sb", "sytrd_sb2st",
            "sytd2", "sbtrd", "sptrd",
            "sb2st_kernels", "latrd",
            "orgtr", "ormtr", "opgtr", "opmtr",
            "disna", "lae2", "laev2",
            "lagtf", "lagts",
        ]),
        "generalized-computational": ("Generalized Computational", [
            "sygst", "sygs2", "sbgst", "spgst",
            "lag2", "pbstf",
        ]),
        "tridiag-dc": ("Tridiagonal Divide-and-Conquer", [
            "laed0", "laed1", "laed2", "laed3", "laed4",
            "laed5", "laed6", "laed7", "laed8", "laed9",
            "laeda", "lamrg",
        ]),
        "tridiag-rrr": ("Tridiagonal RRR (MRRR)", [
            "lar1v", "larra", "larrb", "larrc", "larrd",
            "larre", "larrf", "larrj", "larrk", "larrr", "larrv",
        ]),
        "tridiag-bisection": ("Tridiagonal Bisection", [
            "laebz", "laneg",
        ]),
    }),
    "eigenvalues-hermitian": ("Hermitian Eigenvalues", {
        "dense-drivers": ("Dense Matrix Drivers", [
            "heev", "heev_2stage",
            "heevd", "heevd_2stage",
            "heevr", "heevr_2stage",
            "heevx", "heevx_2stage",
        ]),
        "band-drivers": ("Band Matrix Drivers", [
            "hbev", "hbev_2stage",
            "hbevd", "hbevd_2stage",
            "hbevx", "hbevx_2stage",
        ]),
        "packed-drivers": ("Packed Matrix Drivers", [
            "hpev", "hpevd", "hpevx",
        ]),
        "generalized-drivers": ("Generalized Eigenvalue Drivers", [
            "hegv", "hegv_2stage", "hegvd", "hegvx",
            "hbgv", "hbgvd", "hbgvx",
            "hpgv", "hpgvd", "hpgvx",
        ]),
        "reduction": ("Reduction to Tridiagonal", [
            "hetrd", "hetrd_2stage", "hetrd_he2hb", "hetrd_hb2st",
            "hetd2", "hbtrd", "hptrd",
            "hb2st_kernels",
            "ungtr", "unmtr", "upgtr", "upmtr",
            "laqhb",
        ]),
        "generalized-computational": ("Generalized Computational", [
            "hegst", "hegs2", "hbgst", "hpgst",
        ]),
    }),
    "svd": ("Singular Value Decomposition", {
        "standard-drivers": ("Standard SVD Drivers", [
            "gesvd", "gesdd", "gesvdx", "gesvdq",
            "gejsv", "gesvj",
            "bdsdc", "bdsqr", "bdsvdx",
        ]),
        "generalized-drivers": ("Generalized SVD Drivers", [
            "ggsvd3",
        ]),
        "computational": ("Bidiagonal Reduction", [
            "gebrd", "gebd2", "gbbrd",
            "orgbr", "ormbr", "ungbr", "unmbr",
            "labrd",
            "las2", "lasv2", "lartgs",
            "gsvj0", "gsvj1",
        ]),
        "generalized-computational": ("Generalized SVD Computational", [
            "ggsvp3", "lags2", "lapll", "tgsja",
        ]),
        "bidiag-dc": ("Bidiagonal Divide-and-Conquer", [
            "lasd0", "lasd1", "lasd2", "lasd3", "lasd4",
            "lasd5", "lasd6", "lasd7", "lasd8",
            "lasda", "lasdq", "lasdt",
        ]),
        "bidiag-qr": ("Bidiagonal QR Iteration", [
            "lasq1", "lasq2", "lasq3", "lasq4", "lasq5", "lasq6",
        ]),
    }),
    "blas-like": ("BLAS-like Extensions", {
        "initialize-copy": ("Initialize, Copy, Convert", [
            "_lag2_", "_lat2_", "lag2d", "lag2c", "lat2c",
            "lacpy", "lacp2", "larnv", "laruv", "laset",
            "tfttp", "tfttr", "tpttf", "tpttr", "trttf", "trttp",
        ]),
        "vector-ops": ("Vector Operations", [
            "lasrt", "rscl", "drscl",
            "lacgv", "sum1", "max1",
            "larscl2", "lascl2",
        ]),
        "matrix-vector-ops": ("Matrix-Vector Operations", [
            "ilalc", "ilalr", "lascl",
            "symv", "syr", "spmv", "spr",
        ]),
        "matrix-matrix-ops": ("Matrix-Matrix Operations", [
            "sfrk", "hfrk", "lagtm", "tfsm",
            "lacrm", "larcm",
        ]),
        "norms": ("Matrix Norms", [
            "langb", "lange", "langt",
            "lansb", "lansy", "lansf", "lansp",
            "lanhe", "lanhb", "lanhf", "lanhp", "lanht",
            "lanhs", "lanst",
            "lantb", "lantp", "lantr",
            "lassq",
        ]),
        "scalar-ops": ("Scalar Operations", [
            "lamch", "isnan", "ladiv", "ladiv1", "ladiv2", "laisnan",
            "lapy2", "lapy3", "larmm",
        ]),
    }),
    "auxiliary": ("Auxiliary Parameters", [
        "xerbla", "ieeeck", "ilaenv2stage", "iparam2stage", "labad",
    ]),
}


# ---------------------------------------------------------------------------
# Source lookup
# ---------------------------------------------------------------------------
def get_sources(page_name, prefix, src_dir):
    """Return list of source basenames (without .c) that exist for a page.

    For standard pages: tries <prefix><page_name>.c in src_dir.
    For SPECIAL_SOURCES: uses the explicit mapping.
    For PRECISION_INDEPENDENT: returns the bare filename under 'd' only.
    """
    if page_name in PRECISION_INDEPENDENT:
        # Precision-independent: only emit once (under 'd' precision)
        if prefix == "d":
            path = os.path.join(SRC_AUX, f"{page_name}.c")
            if os.path.exists(path):
                return [page_name]
            # Fallback to src/d for files not yet moved
            path = os.path.join(SRC_D, f"{page_name}.c")
            if os.path.exists(path):
                return [page_name]
        return []

    if page_name in SPECIAL_SOURCES:
        spec = SPECIAL_SOURCES[page_name]
        if prefix not in spec:
            return []
        return [f for f in spec[prefix]
                if os.path.exists(os.path.join(src_dir, f"{f}.c"))]

    # Standard: prefix + page_name
    fname = f"{prefix}{page_name}"
    if os.path.exists(os.path.join(src_dir, f"{fname}.c")):
        return [fname]
    return []


# ---------------------------------------------------------------------------
# RST generation helpers
# ---------------------------------------------------------------------------
def make_doxygenfile_block(filename, indent):
    """Generate a doxygenfile directive block."""
    return (f"{indent}.. doxygenfile:: {filename}\n"
            f"{indent}   :project: semicolon-lapack\n"
            f"{indent}   :sections: func")


def generate_leaf_rst(page_name):
    """Generate RST content for a leaf (routine) page."""
    title = page_name
    underline = "=" * len(title)

    # Precision-independent: bare doxygenfile, no tabs
    if page_name in PRECISION_INDEPENDENT:
        return (f"{title}\n{underline}\n\n\n"
                f"{make_doxygenfile_block(page_name + '.c', '')}\n\n")

    # Find which precisions have source files
    available = []
    for prefix, label, sync_key, src_dir in PRECISIONS:
        sources = get_sources(page_name, prefix, src_dir)
        if sources:
            available.append((prefix, label, sync_key, sources))

    if not available:
        return (f"{title}\n{underline}\n\n"
                f".. note:: No source files found for ``{page_name}``.\n")

    if len(available) == 1:
        # Single precision: bare doxygenfile(s), no tabs
        _, _, _, sources = available[0]
        lines = [title, underline, "", ""]
        for src in sources:
            lines.append(make_doxygenfile_block(f"{src}.c", ""))
            lines.append("")
        return "\n".join(lines) + "\n"

    # Multiple precisions: tab-set
    is_multi = any(len(srcs) > 1 for _, _, _, srcs in available)
    lines = [title, underline, "", ".. tab-set::", ""]

    for prefix, label, sync_key, sources in available:
        if is_multi:
            lines.append(f"    .. tab-item:: {label}")
        else:
            func_name = sources[0]
            lines.append(f"    .. tab-item:: {func_name}")
            lines.append(f"        :name: {func_name}")
        lines.append(f"        :sync: {sync_key}")
        lines.append("")
        for src in sources:
            lines.append(make_doxygenfile_block(f"{src}.c", "        "))
            lines.append("")

    return "\n".join(lines) + "\n"


def generate_toctree_rst(title, entries):
    """Generate RST content for a toctree (intermediate) page."""
    lines = [title, "=" * len(title), "", ".. toctree::", "   :maxdepth: 1", ""]
    for entry in entries:
        lines.append(f"   {entry}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File I/O with change tracking
# ---------------------------------------------------------------------------
def write_file(filepath, content, dry_run=False, stats=None):
    """Write content to file if changed. Returns True if file was updated."""
    if os.path.exists(filepath):
        with open(filepath) as f:
            old = f.read()
        if old == content:
            if stats is not None:
                stats["unchanged"] += 1
            return False

    if dry_run:
        if stats is not None:
            stats["would_update"] += 1
        return True

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)
    if stats is not None:
        stats["updated"] += 1
    return True


def clean_leaf_dir(leaf_dir, expected_files, dry_run=False):
    """Remove stale .rst files in a leaf directory."""
    removed = []
    if not os.path.isdir(leaf_dir):
        return removed
    for fname in os.listdir(leaf_dir):
        if fname.endswith(".rst") and fname not in expected_files:
            fpath = os.path.join(leaf_dir, fname)
            removed.append(fpath)
            if not dry_run:
                os.remove(fpath)
    return removed


# ---------------------------------------------------------------------------
# Hierarchy walker
# ---------------------------------------------------------------------------
def process_node(parent_dir, node_name, node_data, dry_run, stats):
    """Recursively process one node of the hierarchy.

    Returns the toctree entry name for the parent.
    """
    title = node_data[0]
    children = node_data[1]
    node_dir = os.path.join(parent_dir, node_name)
    toctree_path = os.path.join(parent_dir, f"{node_name}.rst")

    os.makedirs(node_dir, exist_ok=True)

    if isinstance(children, list):
        # Leaf container: directory of routine pages
        expected = {f"{page}.rst" for page in children}
        stale = clean_leaf_dir(node_dir, expected, dry_run)
        for f in stale:
            rel = os.path.relpath(f, API_DIR)
            print(f"  {'Would remove' if dry_run else 'Removed'}: {rel}")

        toctree_entries = []
        for page_name in children:
            leaf_path = os.path.join(node_dir, f"{page_name}.rst")
            content = generate_leaf_rst(page_name)
            write_file(leaf_path, content, dry_run, stats)
            toctree_entries.append(f"{node_name}/{page_name}")

        toctree_content = generate_toctree_rst(title, toctree_entries)
        write_file(toctree_path, toctree_content, dry_run, stats)

    elif isinstance(children, dict):
        # Intermediate node: sub-families
        sub_entries = []
        for sub_name, sub_data in children.items():
            process_node(node_dir, sub_name, sub_data, dry_run, stats)
            sub_entries.append(f"{node_name}/{sub_name}")

        toctree_content = generate_toctree_rst(title, sub_entries)
        write_file(toctree_path, toctree_content, dry_run, stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate API documentation structure with precision tabs"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without writing files"
    )
    args = parser.parse_args()

    print("Source directories:", file=sys.stderr)
    for prefix, label, sync_key, src_dir in PRECISIONS:
        exists = os.path.isdir(src_dir)
        count = (len([f for f in os.listdir(src_dir) if f.endswith(".c")])
                 if exists else 0)
        status = f"{count} files" if exists else "not found"
        print(f"  {label:16s} ({prefix}) : {status}", file=sys.stderr)
    print(file=sys.stderr)

    stats = {"updated": 0, "unchanged": 0, "would_update": 0}
    dry_run = args.dry_run

    # Count leaf pages in hierarchy
    def count_pages(node):
        children = node[1]
        if isinstance(children, list):
            return len(children)
        return sum(count_pages(v) for v in children.values())

    total_pages = sum(count_pages(v) for v in HIERARCHY.values())

    # Process hierarchy
    top_entries = []
    for family_name, family_data in HIERARCHY.items():
        process_node(API_DIR, family_name, family_data, dry_run, stats)
        top_entries.append(family_name)

    # Write api/index.rst
    index_content = generate_toctree_rst("API Reference", top_entries)
    write_file(os.path.join(API_DIR, "index.rst"), index_content, dry_run, stats)

    # Report
    action = "Would update" if dry_run else "Updated"
    n_changed = stats["would_update"] if dry_run else stats["updated"]
    print(f"Hierarchy: {total_pages} leaf pages", file=sys.stderr)
    print(f"  {action}: {n_changed}", file=sys.stderr)
    print(f"  Unchanged: {stats['unchanged']}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
