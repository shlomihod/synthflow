# Configuration file for the Sphinx documentation builder.

from datetime import date

import synthflow

# -- Project information

project = "synthflow"
author = "Shlomi Hod"
year = date.today().year
copyright = "2024-{}, {}".format(year, author)

_full_version = synthflow.__version__
release = _full_version.split("+", 1)[0]
version = ".".join(release.split(".")[:2])

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

# html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"

# -- Options for AutoAPI

autoapi_type = "python"
autoapi_dirs = ["../synthflow"]
autodoc_typehints = "description"
