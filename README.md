![StruM logo](docs/img/StruM_logo.png)

## StruM: Structural Motifs

The interaction between sequence specific transcription factors and DNA requires compatability at a complex interface of both electrostatics and 3D structure. These features depend on the sequence context of the motif, with dependence between adjacent nucleotides. 

StruMs use estimates of DNA shape to model binding sites for sequence specific transcription factors. Here we provide tools to:

+ [Train models](examples/basic.py) from known binding sites
+ Use expectation maximization to [find _de novo_ motifs](examples/em.py)
+ Provide utility to [incorporate additional quantitative features](examples/DNase.py), _e.g._ DNase hypersensitivity.
+ [Score matches](examples/basic.py) to novel sequences

For full documentation, refer to the docs [here (html)](https://pdeford.github.io/StructuralMotifs/) or [here (pdf)](https://github.com/pdeford/StructuralMotifs/raw/master/docs/_build/latex/StructuralMotifs.pdf).

### Installation

Ensure that you have all of the dependencies, and then run:

```
wget "https://github.com/pdeford/StructuralMotifs/archive/master.zip"
unzip -q master.zip
cd StructuralMotifs-master
```

If you only want it installed into your current directory for small scale
use, use the following command::

```
python setup.py build_ext --inplace
```

If you want to install it system wide, use this command instead::

```
python setup.py build_ext
python setup.py install
```

**Dependencies**

+ [numpy](http://www.numpy.org/)
+ [scipy](https://www.scipy.org/)
+ [matplotlib](https://matplotlib.org/)
+ [cython](https://cython.org/)
+ [openmp](https://www.openmp.org) compatible compiler

**With conda**

```
conda install numpy scipy matplotlib cython openmp
```
