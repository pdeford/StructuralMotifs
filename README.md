![StruM logo](docs/img/StruM_logo.png)

## StruM: Structural Motifs

The interaction between sequence specific transcription factors and DNA requires compatability at a complex interface of both electrostatics and 3D structure. These features depend on the sequence context of the motif, with dependence between adjacent nucleotides. 

StruMs use estimates of DNA shape to model binding sites for sequence specific transcription factors. Here we provide tools to:

+ [Train models](examples/basic.py) from known binding sites
+ Use expectation maximization to [find _de novo_ motifs](examples/em.py)
+ Provide utility to [incorporate additional quantitative features](examples/DNase.py), _e.g._ DNase hypersensitivity.
+ [Score matches](examples/basic.py) to novel sequences

For full documentation, refer to the docs [here (html)](https://pdeford.github.io/StructuralMotifs/) or [here (pdf)](docs/_build/latex/StructuralMotifs.pdf).

### Installation

Ensure that you have all of the dependencies, and then run:

```
python setup.py install
```

**Dependencies**

+ [numpy](http://www.numpy.org/)
+ [pandas](https://pandas.pydata.org/)
+ [scipy](https://www.scipy.org/)
+ [matplotlib](https://matplotlib.org/)
+ [bx-python](https://github.com/bxlab/bx-python)

**With PyPI**

```
pip install numpy scipy pandas matplotlib bx-python
```

**With conda**

```
conda install numpy scipy pandas matplotlib
conda install -c bioconda bx-python
```
