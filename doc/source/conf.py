"""
Sphinx configuration for semicolon-lapack documentation.

Uses Doxygen + Breathe for C API documentation.
"""

import os
import sys
from datetime import datetime

from pygments.lexer import inherit
from pygments.lexers.c_cpp import CLexer
from pygments.token import Keyword


class SemicolonCLexer(CLexer):
    """C lexer extended with semicolon-lapack type aliases."""
    name = 'SemicolonC'
    aliases = ['semicolon-c']
    tokens = {
        'statements': [
            (r'\b(INT|i32|i64|f32|f64|c64|c128)\b', Keyword.Type),
            inherit,
        ],
    }

# Add custom extensions directory to path
sys.path.insert(0, os.path.abspath('_ext'))

# -- Project information -----------------------------------------------------

project = 'semicolon-lapack'
author = 'semicolon-lapack developers'
copyright = f'{datetime.now().year}, {author}'

version = '0.1'
release = '0.1.0-dev'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.mathjax',      # LaTeX math rendering
    'sphinx.ext.intersphinx',  # Link to other projects
    'sphinx.ext.todo',         # TODO directives
    'sphinx_copybutton',       # Copy button for code blocks
    'sphinx_design',           # Tabs, cards, grids
    'sig_formatter',           # Custom C function signature formatter
    'param_formatter',         # Custom parameter list formatter
]

# Try to load Breathe for C API docs
try:
    import breathe
    extensions.append('breathe')
    has_breathe = True
except ImportError:
    print("Warning: Breathe not found. C API documentation will be unavailable.")
    print("Install with: pip install breathe")
    has_breathe = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The master toctree document
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'

html_title = f"{project} v{version}"
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['tab-deeplink.js']

html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "search_bar_text": "Search functions...",
    "github_url": "https://github.com/ilayn/semicolon-lapack",
}

# -- Breathe configuration ---------------------------------------------------

if has_breathe:
    # Path must be relative to conf.py location or absolute
    _doc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    breathe_projects = {
        'semicolon-lapack': os.path.join(_doc_dir, 'build', 'doxygen', 'xml')
    }
    breathe_default_project = 'semicolon-lapack'
    breathe_default_members = ('members', 'undoc-members')

    # Domain for C code
    breathe_domain_by_extension = {
        "h": "c",
        "c": "c",
    }

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# -- MathJax configuration ---------------------------------------------------

mathjax3_config = {
    'tex': {
        'macros': {
            'mat': [r'\mathbf{#1}', 1],  # \mat{A} for matrices
            'vec': [r'\mathbf{#1}', 1],  # \vec{x} for vectors
            'T': r'^\mathsf{T}',         # \T for transpose
        }
    }
}

# -- Todo extension configuration --------------------------------------------

todo_include_todos = True

# -- Suppress warnings for C types -------------------------------------------

# Common C types that Sphinx doesn't recognize
nitpick_ignore = [
    ('c:identifier', 'FILE'),
    ('c:identifier', 'size_t'),
    ('c:identifier', 'INT'),
    ('c:identifier', 'i32'),
    ('c:identifier', 'i64'),
    ('c:identifier', 'int64_t'),
    ('c:identifier', 'int32_t'),
    ('c:identifier', 'f32'),
    ('c:identifier', 'f64'),
    ('c:identifier', 'c64'),
    ('c:identifier', 'c128'),
]

# -- Copy button configuration -----------------------------------------------

# Don't copy shell prompts
copybutton_prompt_text = r"^\$ |^>>> |^In \[\d+\]: "
copybutton_prompt_is_regexp = True

# -- Custom C lexer with project type aliases --------------------------------

from sphinx.highlighting import lexers
lexers['c'] = SemicolonCLexer()
