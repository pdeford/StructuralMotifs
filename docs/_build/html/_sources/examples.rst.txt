========================================================================
Examples
========================================================================

This page contains simple examples showcasing the basic functionality
of the StruM package.

.. contents::


------------------------------------------------------------------------
Basic Example
------------------------------------------------------------------------

The basic usage of the StruM package covers the case where the binding
site is already known. If the PWM is known already, a tool such as `FIMO
<http://meme-suite.org/doc/fimo.html>`_ could be used to determine 
possible matches to the binding site. 

In this case, a maximum likelihood StruM can be computed directly from
these sequences. Source code for this example can be found here: 
`basic.py <https://github.com/pdeford/StructuralMotifs/blob/master/examples/basic.py>`_.

.. literalinclude:: ../examples/basic.py
   :language: python
   :linenos:

You should see the position weight matrix that would be derived
from these sequences followed by the strand, position, matching 
kmer, and score for the best match in the test sequence to the
StruM. The labels surrounding the PWM can be toggled off by 
changing the ``labels`` parameter in :func:`strum.StruM.print_PWM`. ::

        1     2     3     4     5     6     7     8     9    10    11
    A 0.000 0.222 0.333 0.444 0.000 0.000 1.000 1.000 1.000 0.000 1.000
    C 1.000 0.000 0.000 0.000 0.000 0.778 0.000 0.000 0.000 0.889 0.000
    G 0.000 0.000 0.667 0.000 1.000 0.000 0.000 0.000 0.000 0.000 0.000
    T 0.000 0.778 0.000 0.556 0.000 0.222 0.000 0.000 0.000 0.111 0.000

    + 8 CAGAGCAAACA -22.2070756787

------------------------------------------------------------------------
*De Novo* Motif Finding Example
------------------------------------------------------------------------

More often, the task is to take a set of sequences and perform *de novo*
motif finding, simultaneously modeling the motif and identifying the
putative binding sites. The StruM package allows you to do this using
expectation maximization.

The following example uses the EM capabilities of the StruM package to
identify the motif GATTACA randomly inserted into 50 sequences. The
source gode for this example can be downloaded here: `em.py <https://github.com/pdeford/StructuralMotifs/blob/master/examples/em.py>`_. 

Refer to :func:`strum.StruM.train_EM` for more info on the parameters for the EM
algorithm.

.. literalinclude:: ../examples/em.py
   :language: python
   :linenos:

There are several sections that are reported in the output. First, the StruM
gives a live update on how many iterations it took to converge for each of
the random restarts. Once they have all completed, it shows the likelihoods
for each of the restarts. The example output below highlights why it is
important to repeat this process a number of times, as the results are
highly variable. Finally we output a summary of the results. It is obvious
that the model correctly identified the sequence NGATTACA as observed both
in the highlighted sequences and the summary PWM at the bottom. ::

    Retaining 50 out of 50 sequences, based on length (>25bp)
    Converged after 11 iterations on likelihood
    Converged after 6 iterations on likelihood
    ...
    Restart Likelihoods: [-6048.060534107214, -6048.060534107214, -6708.26909246273, -6775.863585657122, -6780.922068576558, -6785.073584457321, -6792.228306613402, -6826.92034955017, -6859.951259179826, -6859.951259179826, -6859.951259179827, -6862.955165236004, -6862.955165236007, -6907.68229668809, -6984.121244263931, -6984.121244263931, -7028.366606538122, -7037.3445292597635, -7037.344529511228, -7154.7520543497985]
    +0                    CGATTACA gatctcccgcgaccctt
    +2                 ga TGATTACA agatgcgtcgaatat
    +12      agccctgtccgc AGATTACA accac
    +12      agttaactccct AGATTACA tttgt
    ...
    [ 0.26  0.    1.    0.    0.    1.    0.    1.  ]
    [ 0.28  0.    0.    0.    0.    0.    1.    0.  ]
    [ 0.24  1.    0.    0.    0.    0.    0.    0.  ]
    [ 0.22  0.    0.    1.    1.    0.    0.    0.  ]

------------------------------------------------------------------------
Adding Additional Features
------------------------------------------------------------------------

With the StruM package, in additional to the structural features 
provided in the DiProDB table, you can incorporate additional arbitrary
features, as long as they provide a quantitative value across the
binding site.

In the example below we will define a function that looks up a DNase
signal from a ``bigwig`` file to incorporate. The file we will be using
comes from a DNase experiment in K562 cells, mapped to *hg19* from the
ENCODE project (`ENCFF111KJD <https://www.encodeproject.org/files/ENCFF111KJD/>`_) 
and can be downloaded from `here <https://www.encodeproject.org/files/ENCFF111KJD/@@download/ENCFF111KJD.bigWig>`_.

The source code for this example can be found here: `DNase.py <https://github.com/pdeford/StructuralMotifs/blob/master/examples/DNase.py>`_.

.. literalinclude:: ../examples/DNase.py
   :language: python
   :linenos:

This CTCF site is correctly identified as being in the forward 
orientation, and there is an additional feature being considered. ::

    ['Twist', 'Rise', 'Bend']
    ['Twist', 'Rise', 'Bend', 'k562_DNase']
    Best match found on the '+' strand

------------------------------------------------------------------------
Plotting a graphical representation of the StruM
------------------------------------------------------------------------

You may find it useful to look at a graphical representation of the
motif learned using the methods above. This can be accomplished useing
the :func:`strum.StruM.plot` method. Each of the features included in
the StruM will be displayed as its own linegraph. The line represents
the average value of that feature across the site, and the shaded area
represents +/- 1 standard deviation.

.. literalinclude:: ../examples/plot.py
   :language: python
   :linenos:

This code produces the image below. If you consider the Major Groove 
Distance, you will notice that there is more variation at the 
beginning of the motif (to the left) than at the end, as indicated by
the shaded region. It is also clear to see that FOXA1 prefers a higher
value at position 6, and a low value at position 9.

.. image:: img/strumplot.png