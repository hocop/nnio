import nnio

# -- Project information -----------------------------------------------------
project = 'nnio'
copyright = '2021, Ruslan Baynazarov'
author = 'Ruslan Baynazarov'
version = nnio.__version__
# release = nnio.__version__

# -- Extensions --------------------------------------------------------------
extensions = ['sphinx.ext.autodoc']
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'