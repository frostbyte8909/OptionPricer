from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "optionpricer.models._binomial_cy",
        ["optionpricer/models/_binomial_cy.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    name="optionpricer",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)
