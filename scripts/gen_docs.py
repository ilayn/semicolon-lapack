#!/usr/bin/env python3
"""Generate precision-tabbed RST documentation from existing API reference files.

Usage:
    python scripts/gen_docs.py              # update all RST files
    python scripts/gen_docs.py --dry-run    # preview without writing
    python scripts/gen_docs.py --file hesv.rst           # single file
    python scripts/gen_docs.py --file hesv.rst --dry-run # preview single file

Reads doc/source/api/**/*.rst, finds `.. doxygenfile:: dXXX.c` directives,
checks which precision variants (s, c, z) exist in src/, and generates
sphinx-design tab-set blocks for multi-precision pages.

No real↔complex name mapping (sy→he, or→un, etc.) is applied. The script
does a trivial prefix swap (d→s, d→c, d→z) and relies on file-existence
checks. Routines where complex names differ (symmetric/orthogonal families)
will only show real-precision tabs; complex-only families get separate pages.
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
DOC_API = os.path.join(PROJECT_ROOT, "doc", "source", "api")
SRC_D = os.path.join(PROJECT_ROOT, "src", "d")
SRC_S = os.path.join(PROJECT_ROOT, "src", "s")
SRC_C = os.path.join(PROJECT_ROOT, "src", "c")
SRC_Z = os.path.join(PROJECT_ROOT, "src", "z")

# Precision definitions: (prefix, label, sync_key, src_dir)
PRECISIONS = [
    ("d", "Double",         "double",         SRC_D),
    ("s", "Single",         "single",         SRC_S),
    ("z", "Complex Double", "complex-double", SRC_Z),
    ("c", "Complex Single", "complex-single", SRC_C),
]

# Regex to find doxygenfile directives referencing d-prefixed C files
RE_DOXYGENFILE_D = re.compile(r"\.\.\s+doxygenfile::\s+(d\w+\.c)")


# ---------------------------------------------------------------------------
# Derive precision variants
# ---------------------------------------------------------------------------
def derive_filename(d_filename, prefix):
    """Derive a precision variant filename via simple prefix swap.

    E.g., derive_filename("dgetrf.c", "s") -> "sgetrf.c"
    """
    return prefix + d_filename[1:]


def find_available_precisions(d_filenames):
    """For a list of d-prefixed filenames, find which precisions have ALL files.

    Returns list of (prefix, label, sync_key, [filenames]) for precisions
    where every derived file exists. Double is always included (source of truth).
    """
    available = []
    for prefix, label, sync_key, src_dir in PRECISIONS:
        if prefix == "d":
            # Double always available (it's what the RST already references)
            available.append((prefix, label, sync_key, list(d_filenames)))
            continue
        derived = [derive_filename(f, prefix) for f in d_filenames]
        if all(os.path.isfile(os.path.join(src_dir, f)) for f in derived):
            available.append((prefix, label, sync_key, derived))
    return available


# ---------------------------------------------------------------------------
# RST generation
# ---------------------------------------------------------------------------
def make_doxygenfile_block(filename, indent):
    """Generate a doxygenfile directive block."""
    lines = []
    lines.append(f"{indent}.. doxygenfile:: {filename}")
    lines.append(f"{indent}   :project: semicolon-lapack")
    lines.append(f"{indent}   :sections: func")
    return "\n".join(lines)


def make_tab_item(label, sync_key, filenames, name=None):
    """Generate a single tab-item block."""
    lines = []
    if name:
        lines.append(f"    .. tab-item:: {label} ({name})")
        lines.append(f"        :name: {name}")
    else:
        lines.append(f"    .. tab-item:: {label}")
    lines.append(f"        :sync: {sync_key}")
    lines.append("")
    for i, fname in enumerate(filenames):
        lines.append(make_doxygenfile_block(fname, "        "))
        if i < len(filenames) - 1:
            lines.append("")
    return "\n".join(lines)


def generate_tabbed_rst(title, underline_char, d_filenames):
    """Generate full RST content with tab-set for multi-precision."""
    available = find_available_precisions(d_filenames)

    if len(available) <= 1:
        # Only double — no tabs needed, generate bare format
        return generate_bare_rst(title, underline_char, d_filenames)

    is_multi = len(d_filenames) > 1
    lines = []
    lines.append(title)
    lines.append(underline_char * len(title))
    lines.append("")
    lines.append(".. tab-set::")
    lines.append("")

    for prefix, label, sync_key, filenames in available:
        # For single-file tabs: include routine name in label and :name:
        if is_multi:
            tab = make_tab_item(label, sync_key, filenames)
        else:
            routine_name = filenames[0].replace(".c", "")
            tab = make_tab_item(label, sync_key, filenames, name=routine_name)
        lines.append(tab)
        lines.append("")

    return "\n".join(lines) + "\n"


def generate_bare_rst(title, underline_char, d_filenames):
    """Generate RST with bare doxygenfile directives (no tabs)."""
    lines = []
    lines.append(title)
    lines.append(underline_char * len(title))
    lines.append("")
    lines.append("")
    for i, fname in enumerate(d_filenames):
        lines.append(make_doxygenfile_block(fname, ""))
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Parse existing RST
# ---------------------------------------------------------------------------
def parse_rst(filepath):
    """Parse an RST file and extract title and d-prefixed doxygenfile refs.

    Returns (title, underline_char, d_filenames) or None if no d-prefixed
    doxygenfile directives found.
    """
    with open(filepath) as f:
        content = f.read()

    # Extract title: first non-blank line followed by underline
    lines = content.split("\n")
    title = None
    underline_char = "="
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and title is None:
            title = stripped
            continue
        if title is not None and stripped:
            # This should be the underline
            if all(c == stripped[0] for c in stripped):
                underline_char = stripped[0]
            break

    if title is None:
        return None

    # Find all d-prefixed doxygenfile references
    d_filenames = RE_DOXYGENFILE_D.findall(content)
    if not d_filenames:
        return None

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in d_filenames:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return (title, underline_char, unique)


# ---------------------------------------------------------------------------
# Process a single RST file
# ---------------------------------------------------------------------------
def process_file(filepath):
    """Process one RST file. Returns (new_content, changed) or (None, False)."""
    parsed = parse_rst(filepath)
    if parsed is None:
        return None, False

    title, underline_char, d_filenames = parsed
    new_content = generate_tabbed_rst(title, underline_char, d_filenames)

    with open(filepath) as f:
        old_content = f.read()

    return new_content, (new_content != old_content)


# ---------------------------------------------------------------------------
# Collect RST files
# ---------------------------------------------------------------------------
def collect_rst_files(root):
    """Recursively collect all .rst files under root."""
    rst_files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if fname.endswith(".rst"):
                rst_files.append(os.path.join(dirpath, fname))
    return sorted(rst_files)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate precision-tabbed RST documentation"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print changes instead of writing files"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Process only this file (basename, e.g. hesv.rst)"
    )
    args = parser.parse_args()

    # Collect files
    if args.file:
        # Find the file in the API doc tree
        matches = []
        for rst in collect_rst_files(DOC_API):
            if os.path.basename(rst) == args.file:
                matches.append(rst)
        if not matches:
            print(f"Error: {args.file} not found under {DOC_API}",
                  file=sys.stderr)
            return 1
        rst_files = matches
    else:
        rst_files = collect_rst_files(DOC_API)

    # Report source directories
    for prefix, label, sync_key, src_dir in PRECISIONS:
        exists = os.path.isdir(src_dir)
        count = len([f for f in os.listdir(src_dir) if f.endswith(".c")]) \
            if exists else 0
        status = f"{count} files" if exists else "not found"
        print(f"  {label:16s} ({prefix}) : {src_dir} ({status})",
              file=sys.stderr)

    updated = 0
    skipped = 0
    unchanged = 0

    for filepath in rst_files:
        relpath = os.path.relpath(filepath, PROJECT_ROOT)
        new_content, changed = process_file(filepath)

        if new_content is None:
            skipped += 1
            continue

        if not changed:
            unchanged += 1
            continue

        if args.dry_run:
            print(f"=== {relpath} ===")
            print(new_content)
        else:
            with open(filepath, "w") as f:
                f.write(new_content)
            updated += 1

    print(f"\nProcessed {len(rst_files)} RST files:", file=sys.stderr)
    print(f"  Updated:   {updated}", file=sys.stderr)
    print(f"  Unchanged: {unchanged}", file=sys.stderr)
    print(f"  Skipped:   {skipped} (no d-prefixed doxygenfile)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
