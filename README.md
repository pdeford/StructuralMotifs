## StruM: Structural Motifs

The interaction between sequence specific transcription factors and DNA requires compatability at a complex interface of both electrostatics and 3D structure. These features depend on the sequence context of the motif, with dependence between adjacent nucleotides. Many sequence based methods are insufficient to capture these characteristics.

StruMs use estimates of DNA shape to model binding sites for sequence specific transcription factors. Here we provide tools to:

+ [Train models](examples/basic.py) from known binding sites
+ Use expectation maximization to [find _de novo_ motifs](examples/em.py)
+ Provide utility to [incorporate additional quantitative features](examples/DNase.py), _e.g._ DNase hypersensitivity.
+ [Score matches](examples/basic.py) to novel sequences

------------------------------------------------------------

### Specification

StruMs take DNA shape estimates from sources such as the [Dinucleotide Property Database (DiProDb)](http://diprodb.leibniz-fli.de/) or [`DNAshapeR`](http://bioconductor.org/packages/release/bioc/html/DNAshapeR.html) and estimate a set of parameters (_phi_) that describe each position-specific shape feature. It makes the assumption that for a given position, each shape feature is normally distributed about some optimum value (_v<sub>j</sub>_).

<!-- ![StruM Math](images/StruM_math.png){:height="70%" width="70%" } -->

A new sequence of the appropriate length can then be scored (_s<sub>i</sub>_) by multiplying the probabilities given by the Normal distribution for each position-specific feature. To avoid underflow, this is computed in _log_ space.
