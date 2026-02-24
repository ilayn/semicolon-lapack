#!/usr/bin/env python3
"""
Migrate test files from hardcoded `int` to the project's `INT` macro for ILP64 support.

This script converts LAPACK integer parameters in test files from `int` to `INT`,
which resolves to `i32` (LP64) or `i64` (ILP64) based on the SEMICOLON_ILP64 flag.

Strategy: AGGRESSIVE conversion. Nearly ALL `int` becomes `INT`.
Only CMocka framework API types stay as `int`:
  - int main(void)
  - static int setup*(void** state)
  - static int teardown*(void** state)
  - static int group_setup*(void** state)
  - static int group_teardown*(void** state)
  - return cmocka_run_group_tests_name(...)
  - xlaenv(int ispec, int nvalue) — tuning API, always small

Pointer star goes on the type side: `INT* kl` not `INT *kl`.
Printf format: cast to (long long) with %lld for INT variables.

Usage:
    python scripts/migrate_tests_ilp64.py verify-header tests/d/testutils/verify.h
    python scripts/migrate_tests_ilp64.py verify-header tests/s/testutils/verify.h
    python scripts/migrate_tests_ilp64.py testutils tests/d/testutils/
    python scripts/migrate_tests_ilp64.py testutils tests/s/testutils/
    python scripts/migrate_tests_ilp64.py tests tests/d/smoke/
    python scripts/migrate_tests_ilp64.py tests tests/d/
    python scripts/migrate_tests_ilp64.py tests tests/s/smoke/
    python scripts/migrate_tests_ilp64.py tests tests/s/
"""

import re
import sys
import os
import argparse


# ============================================================================
# verify.h processing
# ============================================================================

def process_verify_header(filepath):
    """Convert int -> INT in verify.h declaration files.

    Strategy: Replace ALL int with INT, then fix the very few CMocka exceptions.
    Also fix star placement: `int *kl` -> `INT* kl`.
    """
    with open(filepath) as f:
        content = f.read()

    original = content

    # Step 1: Change include
    content = content.replace(
        '#include "semicolon_lapack/types.h"',
        '#include "semicolon_lapack/semicolon_lapack.h"'
    )

    # Step 2: Fix star placement first: `int *` -> `int* ` (but not `int **`)
    # This normalizes everything so subsequent replacements work
    content = re.sub(r'\bint \*(\w)', r'int* \1', content)

    # Step 3: Replace all int variants with INT
    # Order matters: longer patterns first to avoid partial matches

    # const int* -> const INT*
    content = content.replace('const int*', 'const INT*')

    # const int  -> const INT (with variable spacing)
    content = re.sub(r'\bconst int\b', 'const INT', content)

    # int* -> INT*
    content = re.sub(r'\bint\*', 'INT*', content)

    # int followed by identifier (variable decl or param)
    content = re.sub(r'\bint (\w)', r'INT \1', content)

    # int at start of line (return type)
    content = re.sub(r'^int ', 'INT ', content, flags=re.MULTILINE)

    # extern int -> extern INT
    content = re.sub(r'\bextern int\b', 'extern INT', content)

    # int ninfo[2] etc. already handled by `int n` -> `INT n` above

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  Modified: {filepath}")
    else:
        print(f"  Unchanged: {filepath}")


# ============================================================================
# .c file processing (testutils and test drivers)
# ============================================================================

# Lines matching these patterns should NOT have int converted
CMOCKA_LINE_PATTERNS = [
    'int main(',
    'static int setup',
    'static int teardown',
    'static int group_setup',
    'static int group_teardown',
    'return cmocka_run',
    'xlaenv(',
    'xlaenv_reset(',
    'lapack_get_nb(',
]


def remove_cblas_include(content):
    """Remove #include <cblas.h> lines, but only if the file doesn't call CBLAS directly."""
    # If the file calls cblas_* functions, keep the include
    if re.search(r'\bcblas_\w+', content):
        return content
    return re.sub(r'#include\s*<cblas\.h>\s*\n', '', content)


def remove_extern_declarations(content):
    """Remove extern declarations of library routines.

    These multi-line declarations like:
        extern void dgetrf(const int m, const int n, f64* A,
                           const int lda, int* ipiv, int* info);
    are no longer needed because test_harness.h now includes
    semicolon_lapack.h which provides all declarations.
    """
    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip "// Forward declarations" comment before extern block
        if stripped in ('// Forward declarations',
                        '// Forward declaration',
                        '/* Forward declarations */'):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith('extern'):
                i = j
                continue
            else:
                result.append(line)
                i += 1
                continue

        # Detect start of extern declaration for LAPACK routines
        if stripped.startswith('extern ') and re.match(
            r'extern\s+(?:void|f64|f32|c128|c64|INT|int|i32|i64)\s+[dszc]',
            stripped
        ):
            # Multi-line extern: consume until we find );
            extern_block = stripped
            while not extern_block.rstrip().endswith(');'):
                i += 1
                if i >= len(lines):
                    break
                extern_block += ' ' + lines[i].strip()
            i += 1
            # Skip blank line after extern block
            if i < len(lines) and not lines[i].strip():
                i += 1
            continue

        result.append(line)
        i += 1

    return '\n'.join(result)


def is_cmocka_line(stripped):
    """Check if a line is a CMocka API pattern that should stay int."""
    return any(p in stripped for p in CMOCKA_LINE_PATTERNS)


def convert_file_aggressive(content):
    """Aggressively convert int -> INT in a C file.

    Strategy: Process line by line. For each line that is NOT a CMocka pattern,
    replace all `int` occurrences with `INT`.

    This handles:
    - Function signatures: const int m -> const INT m
    - Pointers: int* info -> INT* info, int *info -> INT* info
    - Local variables: int m, n; -> INT m, n;
    - Struct members: int n; -> INT n;
    - Static arrays: static const int MVAL[] -> static const INT MVAL[]
    - Return types: int dgennd(...) -> INT dgennd(...)
    - sizeof: sizeof(int) -> sizeof(INT)
    - Casts: (int) -> (INT)
    """
    lines = content.split('\n')
    result = []

    for line in lines:
        stripped = line.strip()

        # Skip CMocka API lines
        if is_cmocka_line(stripped):
            result.append(line)
            continue

        # Skip preprocessor lines
        if stripped.startswith('#'):
            result.append(line)
            continue

        # Skip lines that don't contain 'int' at all
        if 'int' not in line:
            result.append(line)
            continue

        # Skip lines with uint64_t, int64_t, int32_t, uint32_t etc.
        # We need to be careful not to replace the 'int' inside these types
        # Strategy: temporarily mask these, do replacement, then unmask

        # Mask types that contain 'int' but shouldn't be changed
        masked = line
        masks = []
        mask_counter = 0

        # Mask type names containing 'int': uint64_t, int64_t, int32_t, uint32_t,
        # uint8_t, int8_t, uint16_t, int16_t, uintptr_t, intptr_t, etc.
        # Also: printf, print_error, print_message, snprintf, fprintf, sprint
        # Also: INT (already converted), UINT, HINT etc.
        for pattern in [
            r'\buint\d+_t\b', r'\bint\d+_t\b', r'\buintptr_t\b', r'\bintptr_t\b',
            r'\bINT\b',
            r'\bprintf\b', r'\bsprintf\b', r'\bsnprintf\b', r'\bfprintf\b',
            r'\bprint_error\b', r'\bprint_message\b',
            r'\bstrint\b',  # unlikely but safe
            r'\bpoint\b', r'\bpointer\b', r'\bhint\b',
            r'\binterface\b', r'\binternal\b', r'\binterrupt\b',
            r'\binterval\b', r'\binteger\b', r'\bintegrate\b',
        ]:
            for m in re.finditer(pattern, masked):
                placeholder = f'__MASK{mask_counter}__'
                masks.append((placeholder, m.group()))
                masked = masked[:m.start()] + placeholder + masked[m.end():]
                # Re-find since positions shifted - just do it simply
                break  # Only first match per pattern per iteration

        # Actually, the masking approach above is fragile. Let me use a different
        # strategy: use word boundary regex to only match standalone `int`

        new_line = line

        # Fix star placement first: `int *var` or `int * const` -> `int* var`/`int* const`
        # Consume optional trailing whitespace to avoid double spaces
        new_line = re.sub(r'\bint \*(?!\*)\s*', 'int* ', new_line)

        # Now convert int -> INT, using word boundaries to avoid matching
        # uint64_t, int64_t, printf, etc.

        # sizeof(int) -> sizeof(INT)
        new_line = new_line.replace('sizeof(int)', 'sizeof(INT)')

        # (int) cast -> (INT) cast
        new_line = re.sub(r'\(int\)', '(INT)', new_line)

        # const int* -> const INT*
        new_line = re.sub(r'\bconst int\*', 'const INT*', new_line)

        # const int  -> const INT
        new_line = re.sub(r'\bconst int\b', 'const INT', new_line)

        # int* -> INT*  (but not uint* etc. - \b handles this)
        new_line = re.sub(r'\bint\*', 'INT*', new_line)

        # int followed by space and identifier (variable/param decl)
        # \bint\b matches only standalone 'int', not uint64_t or printf
        new_line = re.sub(r'\bint\b', 'INT', new_line)

        result.append(new_line)

    return '\n'.join(result)


def process_c_file(filepath, mode='testutils'):
    """Process a .c file to convert int -> INT.

    mode: 'testutils' for verification/utility files
          'tests' for test driver files
    """
    with open(filepath) as f:
        content = f.read()

    original = content

    # Step 1: Remove #include <cblas.h>
    content = remove_cblas_include(content)

    # Step 2: Remove extern declarations of LAPACK routines
    content = remove_extern_declarations(content)

    # Step 3: Aggressive int -> INT conversion
    content = convert_file_aggressive(content)

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  Modified: {filepath}")
    else:
        print(f"  Unchanged: {filepath}")


def process_directory(dirpath, mode):
    """Process all .c files in a directory."""
    for filename in sorted(os.listdir(dirpath)):
        if not filename.endswith('.c'):
            continue
        filepath = os.path.join(dirpath, filename)
        if os.path.isfile(filepath):
            process_c_file(filepath, mode=mode)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Migrate test files from int to INT for ILP64 support.'
    )
    parser.add_argument('mode', choices=['verify-header', 'testutils', 'tests'],
                        help='Processing mode')
    parser.add_argument('path', help='File or directory path')

    args = parser.parse_args()

    if args.mode == 'verify-header':
        if not os.path.isfile(args.path):
            print(f"Error: {args.path} is not a file", file=sys.stderr)
            sys.exit(1)
        print(f"Processing verify header: {args.path}")
        process_verify_header(args.path)

    elif args.mode in ('testutils', 'tests'):
        if os.path.isfile(args.path):
            print(f"Processing file: {args.path}")
            process_c_file(args.path, mode=args.mode)
        elif os.path.isdir(args.path):
            print(f"Processing directory: {args.path}")
            process_directory(args.path, mode=args.mode)
        else:
            print(f"Error: {args.path} does not exist", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()
