import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "InvestOS"
copyright = "2023, InvestOS"
author = "Charlie Reese"
release = "0.2.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon"]  # Custom

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_theme_options = {
    # sidebar_collapse: False,
    # sidebar_includehidden: True,
}
html_static_path = ["_static"]

html_css_files = [
    "theme_overrides.css",  # In _static
    "tailwind_output.css",  # In _static
]

html_sidebars = {
    "**": [
        "searchbox.html",
        "relations.html",
        # 'sourcelink.html',
        # 'globaltoc.html',
        # 'localtoc.html',
        "fulltoc.html",
        # 'navigation.html',
    ]
}

# -- Custom configuration ----------------------------------------------------
sys.path.insert(0, os.path.abspath("investos"))

# For more information on below options:
# --> Guide on getting set up (+ RTD dev env): https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/
# --> https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration
# --> https://numpydoc.readthedocs.io/en/latest/format.html
# Linking to python objects in documentation:
# --> https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects

# napoleon_google_docstring = False
# napoleon_use_param = False
# napoleon_use_ivar = True
