# Notes on building documentation from python notebooks and python source files

* to build docs
  * use a virtual environment with sphinx, myst_parser, and nbsphinx installed, then run
    * `cd libraries/src/massspec_ml/pytorch/docs`
    * `sphinx-apidoc -o . ../`
    * `sphinx-build -b html . _build`
    * recursively copy the contents of the _build directory to the destination
* notes
  * source files won't be parsed unless they are in a directory containing __init__.py
  * methods for creating documentation:
    * Each function and class can be documented using restructuredtext placed in the source file.  
      This documentation will automatically appear in the module documentation.
      * examples can be found in our source code and (here)[https://thomas-cokelaer.info/tutorials/sphinx/docstring_python.html],
        with the source code that generated the documentation at the bottom of the page. 
    * jupyter notebooks 
      * the names of the notebooks should be added explicitly by putting their filename without extension into the file
        docs/index.rst underneath "toctree".
       * don't include more than one markdown header per markdown cell.  That is, don't put "## header 1" followed by "## header 2" in the same cell.
    * markdown files
      * the names of the markdown files should be added explicitly by putting their filename without extension into the file docs/index.rst underneath "toctree".
    * restructuredtext (*.rst) files, which are widely used and the same format used to document functions and classes.