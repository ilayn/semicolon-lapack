"""
Sphinx extension to format C function signatures with aligned parameters.

Transforms Breathe's inline signature rendering into a code block with:
- One parameter per line
- Aligned types for easy scanning
- Copy button support via sphinx-copybutton
"""

import re
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.addnodes import desc, desc_signature


def extract_signature_text(sig_node):
    """Extract the raw text from a signature node."""
    return sig_node.astext()


def parse_c_signature(sig_text):
    """
    Parse a C function signature into components.

    Returns: (return_type, func_name, [(leading_const, base_type, qualifiers, name), ...])
    """
    sig_text = sig_text.strip()

    match = re.match(r'^(\w+(?:\s*\*)*)\s+(\w+)\s*\((.*)\)$', sig_text, re.DOTALL)
    if not match:
        return None

    return_type = match.group(1).strip()
    func_name = match.group(2).strip()
    params_str = match.group(3).strip()

    if not params_str or params_str == 'void':
        return (return_type, func_name, [])

    params = []
    param_parts = split_params(params_str)

    for param in param_parts:
        param = param.strip()
        if not param:
            continue

        parsed = parse_param(param)
        if parsed:
            params.append(parsed)

    return (return_type, func_name, params)


def split_params(params_str):
    """Split parameter string by commas, respecting nested parentheses."""
    params = []
    depth = 0
    current = []

    for char in params_str:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == ',' and depth == 0:
            params.append(''.join(current))
            current = []
        else:
            current.append(char)

    if current:
        params.append(''.join(current))

    return params


def parse_param(param_str):
    """
    Parse a single parameter into (leading_const, base_type, qualifiers, name).

    Handles patterns like:
    - const int m           -> ('const', 'int',     '',               'm')
    - const char* norm      -> ('const', 'char*',   '',               'norm')
    - double* const restrict A -> ('',  'double*',  'const restrict', 'A')
    - const double* const restrict A -> ('const', 'double*', 'const restrict', 'A')
    - int* info             -> ('',     'int*',     '',               'info')
    - dselect2_t select     -> ('',     'dselect2_t', '',             'select')
    """
    param_str = param_str.strip()

    tokens = param_str.split()
    if not tokens:
        return None

    if len(tokens) == 1:
        return ('', tokens[0], '', '')

    name = tokens[-1]

    if name.startswith('*'):
        name = name.lstrip('*')
        ptr_count = len(tokens[-1]) - len(name)
        type_tokens = tokens[:-1] + ['*' * ptr_count]
    else:
        type_tokens = tokens[:-1]

    # Normalize: attach '*' to the preceding type word
    normalized = []
    for tok in type_tokens:
        if tok == '*' and normalized:
            normalized[-1] += '*'
        elif tok.startswith('*') and normalized:
            normalized[-1] += '*'
            rest = tok.lstrip('*')
            if rest:
                normalized.append(rest)
        else:
            normalized.append(tok)

    # Extract leading const
    leading_const = ''
    if normalized and normalized[0] == 'const':
        leading_const = 'const'
        normalized = normalized[1:]

    # Find the base type: the token that is a type name (possibly with *)
    # It's the first token that isn't a qualifier keyword
    qualifiers_set = {'const', 'restrict', 'volatile'}
    base_type = ''
    qual_tokens = []

    if normalized:
        base_type = normalized[0]
        qual_tokens = [t for t in normalized[1:] if t in qualifiers_set]

    qualifiers = ' '.join(qual_tokens)

    return (leading_const, base_type, qualifiers, name)


def format_signature(return_type, func_name, params, indent=4):
    """
    Format a parsed signature into four-column aligned multi-line string.

    Columns: [const] [base_type*] [qualifiers] [name]
    Each column is independently padded for visual alignment.
    """
    if not params:
        return f"{return_type} {func_name}(void);"

    max_const = max(len(p[0]) for p in params)
    max_base = max(len(p[1]) for p in params)
    max_qual = max(len(p[2]) for p in params)

    lines = [f"{return_type} {func_name}("]

    for i, (lconst, base, qual, name) in enumerate(params):
        comma = ',' if i < len(params) - 1 else ''
        parts = [' ' * indent]

        if max_const > 0:
            parts.append(lconst.ljust(max_const))
            parts.append(' ')

        parts.append(base.ljust(max_base))

        if max_qual > 0:
            parts.append(' ')
            parts.append(qual.ljust(max_qual))

        parts.append(' ')
        parts.append(f"{name}{comma}")

        lines.append(''.join(parts))

    lines.append(");")

    return '\n'.join(lines)


def format_c_signatures(app, doctree, docname):
    """
    Transform C function signatures in the doctree.

    Finds desc_signature nodes for C functions and replaces the inline
    rendering with a formatted literal_block.
    """
    for desc_node in doctree.traverse(desc):
        if desc_node.get('domain') != 'c':
            continue
        if desc_node.get('objtype') != 'function':
            continue

        for sig_node in desc_node.traverse(desc_signature):
            sig_text = extract_signature_text(sig_node)

            parsed = parse_c_signature(sig_text)
            if not parsed:
                continue

            return_type, func_name, params = parsed
            formatted = format_signature(return_type, func_name, params)

            code_block = nodes.literal_block(formatted, formatted)
            code_block['language'] = 'c'
            code_block['classes'].append('sig-block')

            sig_node.parent.insert(0, code_block)

            sig_node['classes'].append('sig-hidden')


def setup(app: Sphinx):
    """Register the extension with Sphinx."""
    app.connect('doctree-resolved', format_c_signatures)

    app.add_css_file('sig_formatter.css')

    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
