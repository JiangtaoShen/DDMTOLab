# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys

# The package lives under src/, so point autodoc at the checkout rather than
# relying on ddmtolab happening to be installed in the build environment.
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DDMTOLab'
copyright = '2025, Jiangtao Shen'
author = 'Jiangtao Shen'

# Single source of truth for the version: src/ddmtolab/__init__.py
try:
    from ddmtolab import __version__ as release
except ImportError:
    release = '1.0.10'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

# Render numpydoc "Attributes" sections as :ivar: fields. Without this, Napoleon
# emits a second .. attribute:: entry for every field autodoc already documented,
# which Sphinx reports as a duplicate object description.
napoleon_use_ivar = True



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
