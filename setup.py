from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "strum.strum",
        ["strum/strum.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(name='StruMs',
      version='0.3',
      description='Structural Motifs',
      author='Peter DeFord',
      #author_email='pdeford@jhu.edu',
      license = "MIT",
      packages=['strum'],
      package_data={'strum' : ['data/diprodb_2016.txt']},
      url='https://github.com/pdeford/strum',
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy',
          'cython',
          'openmp',
      ],
      ext_modules=cythonize(ext_modules, compiler_directives={'embedsignature': True}),
      include_dirs=[numpy.get_include()]
      # zip_safe=False
)