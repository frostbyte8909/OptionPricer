from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import sys
import os

if sys.platform == "darwin":
    compile_args = ["-Xpreprocessor", "-fopenmp"]
    link_args = ["-lomp"]
    brew_prefix = "/opt/homebrew" if os.path.exists("/opt/homebrew") else "/usr/local"
    omp_includes = [os.path.join(brew_prefix, "opt", "libomp", "include"), np.get_include()]
    omp_libs = [os.path.join(brew_prefix, "opt", "libomp", "lib")]
else:
    compile_args = ["-fopenmp"]
    link_args = ["-fopenmp"]
    omp_includes = [np.get_include()]
    omp_libs = []


extensions = [
    Extension(
        "optionpricer.models._binomial_cy",
        ["optionpricer/models/_binomial_cy.pyx"],
        include_dirs=omp_includes,
        library_dirs=omp_libs,
        extra_compile_args=compile_args,
        extra_link_args=link_args
    ),
    Extension(
        "optionpricer.models._fdm_cy",
        ["optionpricer/models/_fdm_cy.pyx"],
        include_dirs=omp_includes,
        library_dirs=omp_libs,
        extra_compile_args=compile_args,
        extra_link_args=link_args
    )
]

setup(
    name="optionpricer",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)
