========================================================================
Introduction to Structural Motifs
========================================================================

It has long been observed that transcription factors show
a higher affinity to some sequences than others, and seem to tolerate
some variability around these highest-affinity sequences.

The interaction between sequence specific transcription factors and DNA 
requires compatability at a complex interface of both electrostatics and 
3D structure. Analagous to sequence representations of transcription
factor binding sites, we assume that transcription factors have a 
preference for a specific shape of the DNA across the binding site and
will tolerate some variation around that preferred shape.

Representing the values that a given position-specific shape feature may
adopt as :math:`v_j`, we expect the preference of the transcription 
factor to be normally distributed: 

.. math::

   v_j \sim \mathcal{N} (\mu_j,\sigma_j^2 )

In which case, given a set :math:`D` of :math:`n` sequences with 
:math:`m` position-specific features that describe a sample of the 
binding sites preferred by a given transcription factor:

.. math::
   
   D = 
   \begin{Bmatrix}
   	(v_{11}, & v_{12}, & ... & v_{1m}), \\
   	(v_{21}, & v_{22}, & ... & v_{2m}), \\
   	... & ... & ... & ...\\
   	(v_{n1}, & v_{n2}, & ... & v_{nm}), \\
   \end{Bmatrix}

we can compute a set of parameters :math:`\phi` describing the 
specificity of that transcription factor:

.. math::
   \phi = 
   \begin{Bmatrix}
   	(\mu_1,\sigma_1), \\
   	(\mu_2,\sigma_2), \\
   	...\\
   	(\mu_m,\sigma_m)
   \end{Bmatrix}

If we also assume that each feature and each position is independent, 
then calculating the score :math:`s` for the *i*\ th sequence becomes:

.. math::

   s_i = \prod_{j=1}^m P (v_{ij} | \mu_j,\sigma_j^2 )

In order to avoid underflow issues during computation, all 
computations are done in log space.

------------------------------------------------------------------------
*De novo* Motif finding
------------------------------------------------------------------------

For *de novo* motif finding, an expectation maximization approach is 
employed. This approach assumes that there is exactly one occurrence
of the binding site on each of the training sequences. This based on the
OOPS model employed by `MEME <http://meme-suite.org/doc/meme.html>`_.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E-step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The likelihood (:math:`l_{ij}`) of the :math:`j`\ th position in the 
:math:`i`\ th sequence being the start of the binding site is taken to 
be the score of the StruM at that position multiplied by the likelihood 
of the flanking regions matching the background model (:math:`\phi_B`):

.. math::
	l_{ij} = \prod_{n=1}^{j-1}{P(v_{ij}|\phi_B)} \\
		\prod_{n=j}^{j+k-1}{P(v_{ij} | \phi_{i-j+1})} \\
		\prod_{n=j+k}^{N}{P(v_{ij}|\phi_B)}


The likelihoods are then normalized on a by-sequence basis to produce 
:math:`M`, the matrix of expected start positions:

.. math::
	M_{ij} = \frac{l_{ij}}{\sum_{j'=1}^m{l_{ij'}}}


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
M-step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The maximization step takes these likelihoods and calculates maximum 
likelihood values for :math:`\mu` and :math:`\sigma` for each of the 
:math:`m` position-specific features:

.. math::
	\mu_j = \sum_{i=1}^n\sum_{\mathrm{v}} {\frac{v_{ij}
		\cdot M_{ij}}{\sum_{i}\sum_{j}{M_{ij}}}}


.. math::
	\sigma_j = 
	\sum_{i=1}^n\sum_{\mathrm{v}} 
	\frac{(v_{ij} - \mu_j)^2 \cdot M_{ij}}
		 {\sum_{i}\sum_{j}{M_{ij}}
		- \frac{\sum_{i}\sum_{j}{M_{ij}^2}}
		       {\sum_{i}\sum_{j}{M_{ij}}}}

