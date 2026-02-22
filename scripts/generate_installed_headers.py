#!/usr/bin/env python3
"""Generate concrete installed headers with all macros resolved.

Per ILP64_DESIGN.md (lines 371-401): installed public headers have concrete
types baked in — no INT macro, no internal_build_defs.h, no lapack_name_map.h.

LP64 output:  SEMICOLON_API void dgetrf(const i32 m, ...)
ILP64 output: SEMICOLON_API void dgetrf_64(const i64 m, ...)

Usage (called by meson custom_target):
    python tools/generate_installed_headers.py \
        --int-type i32 \
        --naming '' \
        --input-dir src/include \
        --public-dir include/semicolon_lapack \
        --output-dir builddir/installed_headers
"""

import argparse
import os
import re

# Matches: SEMICOLON_API <return_type> <function_name>(
# Same pattern as generate_name_map.py
DECL_RE = re.compile(r"SEMICOLON_API\s+\w+\s+(\w+)\s*\(")

PRECISION_HEADERS = [
    "semicolon_lapack_double.h",
    "semicolon_lapack_single.h",
    "semicolon_lapack_complex_double.h",
    "semicolon_lapack_complex_single.h",
    "semicolon_lapack_auxiliary.h",
]


def extract_function_names(input_dir):
    """Extract all SEMICOLON_API function names from precision headers."""
    names = set()
    for header in PRECISION_HEADERS:
        path = os.path.join(input_dir, header)
        with open(path) as f:
            for line in f:
                m = DECL_RE.search(line)
                if m:
                    names.add(m.group(1))
    return names


def apply_naming(name, pattern):
    """Apply the SYMBOL_MANGLING pattern to a function name.

    Pattern uses C token-pasting syntax: 'name##_64' -> 'dgetrf_64'
    The literal 'name' in the pattern is the placeholder.
    """
    return pattern.replace("##", "").replace("name", name)


def transform_precision_header(content, int_type, func_names, naming,
                               aggregator_name):
    """Transform a precision header to have concrete types."""
    result = content

    # Replace INT with concrete type (word boundary to avoid matching
    # e.g. SEMICOLON_INTERNAL or printf)
    result = re.sub(r"\bINT\b", int_type, result)

    # For ILP64: rename function names and fix aggregator include
    if naming:
        for name in sorted(func_names, key=len, reverse=True):
            new_name = apply_naming(name, naming)
            result = re.sub(r"\b" + re.escape(name) + r"\b", new_name, result)

        # Point precision headers at the renamed aggregator
        result = result.replace(
            '"semicolon_lapack/semicolon_lapack.h"',
            '"semicolon_lapack/' + aggregator_name + '"',
        )

    return result


def generate_aggregator(public_dir, int_type):
    """Generate the installed aggregator (no internal_build_defs.h)."""
    path = os.path.join(public_dir, "semicolon_lapack.h")
    with open(path) as f:
        content = f.read()

    # Remove the internal_build_defs.h include line
    content = re.sub(
        r'#include\s+"internal_build_defs\.h"\s*\n', "", content
    )

    return content


def main():
    parser = argparse.ArgumentParser(
        description="Generate concrete installed headers"
    )
    parser.add_argument("--int-type", required=True, choices=["i32", "i64"])
    parser.add_argument("--naming", required=True,
                        help="SYMBOL_MANGLING pattern (empty for LP64)")
    parser.add_argument("--input-dir", required=True,
                        help="Path to src/include/")
    parser.add_argument("--public-dir", required=True,
                        help="Path to include/semicolon_lapack/")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for generated headers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Aggregator filename: semicolon_lapack_64.h for ILP64, semicolon_lapack.h for LP64
    aggregator_name = "semicolon_lapack_64.h" if args.naming else "semicolon_lapack.h"

    # Extract all function names for renaming
    func_names = extract_function_names(args.input_dir)

    # Generate concrete precision headers
    for header in PRECISION_HEADERS:
        path = os.path.join(args.input_dir, header)
        with open(path) as f:
            content = f.read()

        concrete = transform_precision_header(
            content, args.int_type, func_names, args.naming, aggregator_name
        )

        out_path = os.path.join(args.output_dir, header)
        with open(out_path, "w") as f:
            f.write(concrete)

    # Generate concrete aggregator
    aggregator = generate_aggregator(args.public_dir, args.int_type)
    out_path = os.path.join(args.output_dir, aggregator_name)
    with open(out_path, "w") as f:
        f.write(aggregator)


if __name__ == "__main__":
    main()
