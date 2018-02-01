========================================================================
Installation Instructions
========================================================================

------------------------------------------------------------------------
Manual Installation of Dependencies
------------------------------------------------------------------------

The StruM package depends on the following libraries:

* `numpy <http://www.numpy.org/>`_
* `pandas <https://pandas.pydata.org/>`_
* `scipy <https://www.scipy.org/>`_
* `matplotlib <https://matplotlib.org/>`_
* `bx-python <https://github.com/bxlab/bx-python>`_

------------------------------------------------------------------------
Installing dependencies with Conda
------------------------------------------------------------------------

By far the easiest way to install all of the dependencies is with
`conda <https://conda.io/docs/>`_. With conda installed, the 
dependencies can be installed with::

	conda install numpy scipy pandas matplotlib
	conda install -c bioconda bx-python

------------------------------------------------------------------------
Installing dependencies with PyPI
------------------------------------------------------------------------

Another method for installing the dependencies manually is using the
pip command from `PyPI <https://pypi.python.org/pypi>`_. Then the
dependencies can be installed with::

	pip install numpy scipy pandas matplotlib bx-python

------------------------------------------------------------------------
Installation of StruM Package
------------------------------------------------------------------------

To install the StruM package, simply download the source code, navigate
to that directory, and run the installation::

	wget "https://github.com/pdeford/StructuralMotifs/archive/master.zip"
	unzip -q master.zip
	cd StructuralMotifs-master
	python setup.py install