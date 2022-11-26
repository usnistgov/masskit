from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("search", ["search.pyx"])
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
