"""
Sphinx extension to format Breathe parameter documentation into a styled grid layout.

Transforms Breathe's inline parameter rendering into a 3-column grid:
- Column 1: direction badge (in/out/inout)
- Column 2: parameter name in monospace
- Column 3: description text
"""

import re
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.addnodes import desc, desc_content


def parse_param_entry(list_item):
    """Parse a single parameter list_item into three grid-cell nodes."""
    if not list_item.children:
        return None

    para = list_item.children[0]
    if not isinstance(para, nodes.paragraph):
        return None

    children = list(para.children)
    if len(children) < 2:
        return None

    # child[0]: strong -> parameter name
    # Breathe wraps real parameter names in strong nodes.
    # Reject items where the first child is plain text (e.g. nested
    # description list items like "< 0: iterative refinement failed...").
    if not isinstance(children[0], nodes.strong):
        return None

    param_name = children[0].astext().strip()

    # child[1]: Text " -- " (separator) - skip
    # child[2]: strong "[in]"/"[out]"/"[inout]" (direction) - optional
    direction = ""
    desc_start = 2

    if len(children) >= 3 and isinstance(children[2], nodes.strong):
        dir_text = children[2].astext().strip()
        match = re.match(r'^\[(in|out|inout)\]$', dir_text)
        if match:
            direction = match.group(1)
            desc_start = 3
            # Skip trailing whitespace Text node
            if (desc_start < len(children)
                    and isinstance(children[desc_start], nodes.Text)
                    and not str(children[desc_start]).strip()):
                desc_start += 1

    # Column 1: direction badge
    dir_cell = nodes.container(classes=['param-dir-cell'])
    if direction:
        dir_node = nodes.inline(classes=['param-dir', f'param-dir-{direction}'])
        dir_node += nodes.Text(direction)
        dir_cell += dir_node

    # Column 2: parameter name
    name_cell = nodes.container(classes=['param-name-cell'])
    name_node = nodes.literal(classes=['param-name'])
    name_node += nodes.Text(param_name)
    name_cell += name_node

    # Column 3: description
    desc_cell = nodes.container(classes=['param-desc'])

    # Remaining inline children from the first paragraph
    if desc_start < len(children):
        desc_para = nodes.paragraph()
        for child in children[desc_start:]:
            desc_para += child.deepcopy()
        if desc_para.astext().strip():
            desc_cell += desc_para

    # Additional block-level children of the list_item (e.g., nested lists)
    for extra in list_item.children[1:]:
        desc_cell += extra.deepcopy()

    # Wrap in a display:contents container so all three cells
    # participate in the parent grid
    entry = nodes.container(classes=['param-entry'])
    entry += dir_cell
    entry += name_cell
    entry += desc_cell
    return entry


def format_params(app, doctree, docname):
    """Transform Breathe parameter field_lists into styled param-list containers."""
    for desc_node in doctree.traverse(desc):
        if desc_node.get('domain') != 'c':
            continue
        if desc_node.get('objtype') != 'function':
            continue

        for dc in desc_node.traverse(desc_content):
            for fl in list(dc.children):
                if not isinstance(fl, nodes.field_list):
                    continue

                has_params = False
                for fn in fl.traverse(nodes.field_name):
                    if fn.astext() == 'Parameters':
                        has_params = True
                        break

                if not has_params:
                    continue

                param_list = nodes.container(classes=['param-list'])

                heading = nodes.paragraph(classes=['param-list-heading'])
                heading += nodes.strong(text='Parameters')
                param_list += heading

                for field in fl.traverse(nodes.field):
                    for fb in field.traverse(nodes.field_body):
                        # Only process direct child bullet lists, not nested ones
                        for child in fb.children:
                            if not isinstance(child, nodes.bullet_list):
                                continue
                            for li in child.children:
                                if isinstance(li, nodes.list_item):
                                    entry = parse_param_entry(li)
                                    if entry:
                                        param_list += entry

                fl.replace_self(param_list)


def setup(app: Sphinx):
    """Register the extension with Sphinx."""
    app.connect('doctree-resolved', format_params)
    app.add_css_file('param_formatter.css')

    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
