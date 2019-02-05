# MIT License
#
# Copyright (c) 2019 Peter DeFord
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""StruM: Structural Motifs
----------------------------------------

This package provides functionality for computing structural 
representations of DNA sequence motifs. Estimates for DNA structure
comes from the DiNucleotide Property Database 
(http://diprodb.leibniz-fli.de/).

This version relies on the Cython framework for speed purposes.
Improvements in speed are particularly seen with scoring longer 
sequences.
"""

__version__ = '0.3'
__author__ = 'Peter DeFord'

import cython
from cython.parallel import prange
from cpython cimport bool
cimport openmp

import matplotlib.pyplot as plt
import os
import sys

import numpy as np
cimport numpy as np

###############################################################################

# Pull in math functions from the C library
cdef extern from "math.h":
    double log10(double x) nogil
    double sqrt(double x) nogil
    double erfc(double x) nogil

# Statically type numbers to be used repeatedly
cdef double doubleten = 10.0
cdef double doubleone = 1.0
cdef double M_SQRT1_2 = sqrt(0.5)
cdef double half = 0.5


cdef read_diprodb():
    """Load the values from the DiProDB as a lookup table."""
    cdef list data, features, acids, strands
    cdef dict dinuc_index

    data = []
    features = []
    acids = []
    strands = []
    dipro_path = os.path.join(
        os.path.dirname(__file__), 
        'data/diprodb_2016.txt'
        )
    with open(dipro_path) as f:
        header = f.readline().split("\t")[2:2+16]
        dinuc_index = dict(zip(header, range(16)))
        for line in f:
            fields = line.split("\t")
            features.append(fields[1])
            acids.append(fields[2+16])
            strands.append(fields[2+16+1])
            row = [float(x) for x in fields[2:2+16]]
            data.append(row)

    return np.array(data, dtype=np.double), features, acids, \
           strands, dinuc_index

# Create an index to use for reverse complementing sequences
cdef dict rev_comp_index
nucs = "ACGT"
rev_comp_index = dict(zip(nucs, nucs[::-1]))
rev_comp_index['N'] = 'N'


# Messages for the `train_EM` method.
msg_cnvrg = "Converged after {} iterations on likelihood"
msg_stop = "Stopped after {} iterations"
msg_cycle = "Detected cyclical likelihoods. Proceeding to max."
msg_end = "Did not converge after {} iterations"

class StruM(object):
    """StruM Object: Train and work with Structural Motifs
    
    StruMs can be learned via maximum likelihood (:func:`train`) from a
    known set of aligned binding sites, or via expectation maximization
    (:func:`train_EM`) from a set of broader regions.

    For speed, this package will trim the sequences to be the same
    length. This can be user specified, else the shortest sequence 
    will be used as a guide. 

    Additional features other than those from DiProDB can be 
    incorporated by using the :func:`update` method.
    """

    def __init__(self, str mode="full", int n_process=1, custom_filter=[]):
        """Create a FastStruM object.

        :param mode: Defines which subset of available 
            features in the DiProDB table to use. Choose from: 
            ['basic', 'protein', 'groove', 'proteingroove', 'unique', 
            'full', 'nucs', 'custom']

            +---------------+------------------------------------------+
            | MODE          | Features                                 |
            +===============+==========================================+
            |basic          | Twist, Rise, Bend.                       |
            +---------------+------------------------------------------+
            |protein        | (for DNA-protein complex) Roll, Twist,   |
            |               | Tilt, Slide, Shift, Rise.                |
            +---------------+------------------------------------------+
            |groove         | Major Groove Width, Major Groove Depth,  |
            |               | Major Groove Size, Major Groove Distance,| 
            |               | Minor Groove Width, Minor Groove Depth,  |
            |               | Minor Groove Size, Minor Groove Distance.|
            +---------------+------------------------------------------+
            |proteingroove  | The union of the "protein" and "groove"  |
            |               | filters.                                 |
            +---------------+------------------------------------------+
            |unique         | Filters the table for the first          |
            |               | occurrence of each type of feature.      |
            +---------------+------------------------------------------+
            |full           | All available double stranded DNA        | 
            |               | features.                                |
            +---------------+------------------------------------------+
            |nucs           | Adenine content, Guanine content,        |
            |               | Cytosine content, Thymine content.       |
            +---------------+------------------------------------------+
            |custom         | Manually select desired features.        |
            +---------------+------------------------------------------+
        :type mode: str.
        :param custom_filter: Specifies the indices of 
            desired features from the DiProDB table.
        :type custom_filter: list of ints.
        :param n_process: Number of threads to use. ``-1`` 
            uses all processers.
        :type n_process: int.
        """
        # Dictate how many processes to run. If n == -1,
        # use all the available cores.
        self.n_process = n_process
        if self.n_process == -1:
            self.n_process = openmp.omp_get_num_threads()

        # Read the DiProDB data, and filter the matrix based
        # on the provided mode.
        data, feat, acid, strand, di_index = read_diprodb()
        self.index = di_index
        
        N = data.shape[0]
        masks = {
            "basic"         : [0, 2, 3],
            "groove"        : [6, 7, 8, 9, 10, 11, 12, 13],
            "protein"       : [24, 25, 26, 27, 29, 31],
            "full"          : [i  for i in xrange(N) if \
                               ((acid[i] == "DNA") or \
                                (acid[i] == "B-DNA")) and \
                               (strand[i] == "double")],
            "nucs"          : [78, 79, 80, 81],
            "unique"        : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                               10, 11, 12, 13, 14, 15, 16, 
                               17, 18, 19, 20, 21, 22, 24, 
                               25, 26, 27, 29, 31, 33, 66, 
                               67, 68, 69, 70, 75, 76, 77, 
                               78, 79, 80, 81, 98, 99, 106],
            "proteingroove" : [6, 7, 8, 9, 10, 11, 12, 13] +
                              [24, 25, 26, 27, 29, 31],
            "custom"        : custom_filter,
        }

        assert mode in masks, \
        """Unknown mode: {}
        Please pick from {}""".format(mode, masks.keys())

        self.mode = mode
        self.mask = masks[self.mode]
        self.data = data[masks[mode], :]
        self.p = self.data.shape[0]
        self.features = []
        for i in masks[mode]:
            self.features.append(feat[i])

        # Normalize the DiProDB data
        mean = np.mean(self.data, axis=1)
        sd = np.std(self.data, axis=1)
        self.data = (self.data - mean.reshape([-1,1])) / sd.reshape([-1,1])

        # Initialize attributes for later use
        self.func = None
        self.func_data = None
        self.fit = False
        self.filtered = False
        self.updated = False
        self.filtered = False


    def __str__(self):
        text_vers = self.text_strum(self.strum[0], self.strum[1], colorbar=True)
        return "\n".join(text_vers[::-1])

    def __repr__(self):
        main = """{self.__class__.__name__}(mode="{self.mode}", n_process={self.n_process})
        Attributes:
            k: {self.k}
            p: {self.p}
            fit: {self.fit}
            filtered: {self.filtered}
            updated: {self.updated}
            mask = {self.mask}
            features: {self.features}
        """.format(self=self)
        return main

    def train(self, training_sequences, weights=None, lim=None, **kwargs):
        """Learn structural motif from a set of known binding sites.

        :param training_sequences: Training set, composed of gapless 
            alignment of binding sites of equal length.
        :type training_sequences: list of str. Or if updated, list of tuples: 
            (str, [args])
        :param weights: Weights to associate with each of the sequences
            in ``training_sequences`` to use in learning the motif.
        :type weights: 1D array of floats.
        :param lim: Minimum value allowed for variation in a given
            position-specific-feature. Useful to prevent *any*
            deviation at that position from resulting in a probability of
            0.
        :type lim: float
        
        :return: None. Defines the structural motif ``self.strum`` and the
            corresponding position weight matrix ``self.PWM``, sets attribute
            ``self.fit = True``.
        """
        data = []
        for example in training_sequences:
            data.append(self.translate(example, **kwargs))

        arr = np.asarray(data)
        if weights is None:
            weights = np.ones(arr.shape[0])
        average = np.average(arr, axis=0, weights = weights)
        variance = np.average((arr-average)**2, axis=0, weights = weights)
        self.strum = [average, variance]
        if lim is not None:
            self.strum[1][self.strum[1] < lim] = lim
        self.k = len(training_sequences[0])
        self.define_PWM(training_sequences, weights=weights)
        self.fit = True

    def translate(self, str seq, **kwargs):
        """Convert sequence from string to structural representation.

        :param seq: DNA sequence, all uppercase characters,
            composed of letters from set ACGTN.
        :type seq: str.
        :param \*\*kwargs: Ignored
        
        :return: Sequence in structural representation.
        :rtype: 1D numpy array of floats.
        """
        cdef Py_ssize_t i
        cdef str di
        row = []
        for i in xrange(len(seq)-1):
            di = seq[i:i+2]
            if 'N' in di:
                row.append(np.zeros([self.p,]))
            else:
                row.append(self.data[:, self.index[di]])
        return np.vstack(row).ravel()

    def define_PWM(self, list seqs, weights=None):
        """Computes a position weight matrix from sequences used 
        to train the StruM.

        :param seqs: Training set, composed of gapless 
            alignment of binding sites of equal length.
        :type seqs: list of str.
        :param weights: Weights to associate with each of the sequences
            in ``seqs`` to use in learning the motif.
        :type weights: 1D array of floats.

        :return: None. Sets the position weight matrix ``self.PWM`` 
            based on the weighted sequences.
        """
        cdef dict nuc_index
        cdef str seq, n
        cdef Py_ssize_t i,j
        nuc_index = dict(zip("ACGT", range(4)))
        if weights is None:
            weights = [1.0] * len(seqs)
        pwm = np.zeros([4, self.k])
        for i, seq in enumerate(seqs):
            for j,n in enumerate(seq):
                if n == "N": continue
                pwm[nuc_index[n], j] += weights[i]
        pwm /= np.sum(pwm, axis=0)
        self.PWM = pwm
    def print_PWM(self, labels=False):
        """Pretty prints the PWM to std_out.

        :param labels: Flag indicating whether to print the PWM
            with labels indicating the position associated with 
            each column, and the nucleotide associated with each
            row.
        :type labels: bool.

        :return: Formatted position weight matrix suitable for
            display, or use in the MEME suite, e.g. Also prints
            the PWM to ``std_out``.
        :rtype: str.
        """
        assert (self.fit == True), "No PWM to print. Must call `train` or `train_EM` first."
        nuc_index = dict(zip("ACGT", range(4)))
        rows = [ " ".join(["%0.3f" % x for x in row]) for row in self.PWM ]
        if labels:
            for n in nuc_index:
                rows[nuc_index[n]] = n + " " + rows[nuc_index[n]]
            header = [" ".join([' '*(5-len(x)) + x for x in [str(i+1) for i in range(self.k)]])]
            rows = header + rows
        pretty = "\n".join(rows)
        print pretty
        return pretty

    @cython.boundscheck(False)
    @cython.wraparound(False) 
    def score_seq(self, str seq, **kwargs):
        """Scores a sequence using pre-calculated StruM.

        :param seq: DNA sequence, all uppercase characters,
            composed of letters from set ACGTN.
        :type seq: str.

        :return: Vector of scores for similarity of each kmer
            in ``seq`` to the StruM.
        :rtype: 1D array.
        """
        assert (self.fit == True), \
            "No StruM to score with. Must call `train` or `train_EM` first."
        cdef double val, val2
        cdef Py_ssize_t i, n_kmers, kmr_len, j, kmer_i
        cdef int p = self.p
        cdef int k = self.k
        cdef int n_threads
        cdef double [:] strum_avg = self.strum[0]
        cdef double [:] strum_std = np.zeros(self.strum[1].shape)
        for i in xrange(self.strum[1].shape[0]):
            strum_std[i] = sqrt(self.strum[1][i])

        strucseq = self.translate(seq, **kwargs)
        cdef double [:] strucseq_view = strucseq
        
        n_kmers = len(seq) - k + 1
        kmr_len = p*(k-1)
        
        by_kmer = np.zeros((n_kmers), dtype=np.double)
        cdef double [:] by_kmer_view = by_kmer

        n_threads = self.n_process
        with nogil:
            for i in prange(n_kmers, num_threads=n_threads):
                kmer_i = i*p
                for j in xrange(kmr_len,):
                    val = norm_p(strucseq_view[kmer_i+j], strum_avg[j], strum_std[j])
                    val2 = log10(val)
                    by_kmer_view[i] += val2

        return by_kmer

    def read_FASTA(self, fasta_file):
        """Reads a FASTA formatted file for headers and sequences.

        :param fasta_file: FASTA formatted file containing DNA sequences.
        :type: file object
        
        :return: The headers and sequences from the FASTA file, as two 
            separate lists.
        :rtype: (list, list)
        """
        sequences = []
        headers = []
        header = None
        seq = ""
        for line in fasta_file:
            if line.startswith(">"):
                if header is None:
                    header = line.strip()[1:]
                else:
                    headers.append(header)
                    sequences.append(seq)
                    header = line.strip()[1:]
                    seq = ""
            else:
                seq += line.strip()
        headers.append(header)
        sequences.append(seq)
        return headers, sequences

    @cython.boundscheck(False)
    @cython.wraparound(False) 
    def train_EM(self, data, fasta=True, params=None, int k=10,
        int max_iter=1000, double convergence_criterion=0.001, 
        random_seed=0, int n_init=1, lim=None, seqlength=None,
        background=None, seed_motif=None, bool verbose=False):

        """Performs Expectation-Maximization on a set of sequences 
        to find motif.

        :param data: A set of sequences to use for training the model.
            Assumed to have one occurrence of the binding site per 
            sequence.
        :type data: list of str, open file object referring to a 
            FASTA file.
        :param fasta: Flag indicating whether ``data`` points to an
            open file object containing a FASTA formatted file with
            DNA sequences.
        :type fasta: bool.
        :param params: Additional parameters to pass to ``self.func``,
            if defined.
        :type params: ``*args``, ``**kwargs``.
        :param k: Size of binding site to consider. Since dinucleotides
            are considered, in sequence-space the size of the binding
            site will be ``k + 1``.
        :type k: int.
        :param max_iter: Maximum number of iterations of Expecation
            Maximization to perform if convergence is not attained.
        :type max_iter: int.
        :param convergence_criterion: If the change in the likelihood
            between two iterations is less than this value, the model
            is considered to have converged.
        :type convergence_criterion: float.
        :param random_seed: Seed for the random number generator used
            in the EM algorithm for initialization.
        :type random_seed: int.
        :param n_init: Number of random restarts of the EM algorithm
            to perform.
        :type n_init: int.
        :param lim: Minimum value allowed for variation in a given
            position-specific-feature. Useful to prevent *any*
            deviation at that position from resulting in a probability of
            0.
        :type lim: float
        :param seqlength: If set, the sequences in the training data
            will be trimmed symmetrically to this length. 
            .. note:: This must be longer than the shortes sequence.
        :type seqlength: int.
        :param background: Method to use for computing the background
            model. Default is to assume equal probability of each
            dinucleotide. Passing ``"compute"`` adapts the background
            model to represent the dinucleotide frequencies in the
            training sequences. Otherwise this can the path to a tab 
            delimited file specifying the representation of each 
            dinucleotide. E.g.
            ::
                AA  0.0625
                AC  0.0625
                ...
        :type background: str.
        :param seed_motif: Optional. A `StruM.strum` to use for 
            initializing the Expectation-Maximization algorithm.
            If set, ``k`` will be replaced by the corresponding
            value from the ``seed_motif``'s shape. ``n_init`` will
            also be reset to 1.
        :type seed_motif: ``StruM.strum``.
        :param verbose: Specifies whether to print a text version 
            of the converging StruM at each iteration. Default ``False``.
        :type verboose: bool.

        :return: None. Defines the structural motif ``self.strum`` and the
            corresponding position weight matrix ``self.PWM``.
        """
        err = sys.stderr
        self.k = k + 1

        cdef list sequences
        cdef int n_seqs, n_threads
        cdef Py_ssize_t i, j, ip, jp, jj, ii
        cdef int K = k
        cdef int p = self.p

        # Count threads to use
        n_threads = self.n_process

        # Set the random seed for reproducibility
        if random_seed is not 0:
            np.random.seed(random_seed)

        # Read in the data and define the sequences
        if fasta:
            headers, sequences = self.read_FASTA(data)
        else:
            sequences = data
        sequences_up = []
        n_seqs = len(sequences)
        for i in xrange(n_seqs):
            sequences_up.append(sequences[i].upper())
            sequences_up.append(rev_comp(sequences[i].upper()))

        # Ensure that all of the sequences are the same 
        ##length. If no seqlength is passed, use the length
        ##of the shortest sequence. Otherwise, throw out
        ##sequences shorter than the required length. Select
        ##the subsequence of the appropriate length from the
        ##center of the original sequence.
        # Convert to structural space at the same time.
        if seqlength is None:
            seqlength = min(len(s) for s in sequences)

        cdef list sequences_data = []
        cdef str seq
        cdef int l
        for j in xrange(n_seqs*2):
            seq = sequences_up[j]
            l = len(seq)
            if l >= seqlength:
                i = (l - seqlength)//2
                seq = seq[i:i+seqlength]
                sequences_data.append( self.translate(seq))

        sequences_data_array = np.vstack(sequences_data)
        cdef double [:,:] sequences_data_view = sequences_data_array
        n_seqs = sequences_data_array.shape[0]
        cdef Py_ssize_t N1 = n_seqs
        cdef Py_ssize_t N2 = sequences_data_array.shape[1]
        cdef int nkmers = seqlength - k

        print >> err, "Retaining {} out of {} sequences, based on length (>{}bp)".format(n_seqs/2, len(sequences), seqlength)

        # Precompute the probabilities for each position
        # matching to the background.
        back_s = np.zeros((N1,N2), dtype=np.double)
        all_pos = np.reshape(sequences_data_array, (N1*N2/p, p))
        cdef double [:,:] back_s_view = back_s
        cdef double [:,:] all_pos_view = all_pos

        if background is None:
            # Assume equal representation of each dinucleotide.
            # Results in ~N(0, 1) because that was how the
            # data was normalized
            for i in xrange(N1):
                for j in xrange(N2):
                    back_s_view[i,j] = log10(norm_p(sequences_data_view[i,j], 0., 1.))
        elif background is 'compute':
            # Use the training sequences to compute the background
            # model. Allow every position in every sequence to
            # contribute.
            back_avg = np.average(all_pos, axis=0)
            back_std = np.std(all_pos, axis=0)
            for i in xrange(N1):
                for j in xrange(N2):
                    jp = j%p
                    back_s_view[i,j] = log10(norm_p(sequences_data_array[i,j], back_avg[jp], back_std[jp]))
        else:
            # Read in a precomputed background model. This should
            # be a tab delimited file with two columns -- uppercase
            # dinucleotide followed by its representation in the 
            # organism. E.g.
            # AA    0.0625
            # AC    0.0600
            # AG    0.0650
            # ...
            back_avg = np.zeros(p)
            back_std = np.zeros(p)
            divals = []
            props = []
            with open(background) as backfile:
                for line in backfile:
                    di, prop = line.split()
                    prop = float(prop)
                    di_val = np.reshape(self.translate(di), (p,))
                    divals.append(di_val)
                    props.append(prop)
            weight_sum = np.sum(props)
            props = np.array(props)
            for i,prop in enumerate(props/weight_sum):
                back_avg += prop*divals[i]
            nonzero = np.sum(props != 0)
            weight_sum = (nonzero-1)*np.sum(props)/nonzero
            for i,prop in enumerate(props/weight_sum):
                back_std += prop*(divals[i]-back_avg)
            back_std = np.sqrt(back_std)
            for i in xrange(N1):
                for j in xrange(N2):
                    jp = j%p
                    back_s_view[i,j] = log10(norm_p(sequences_data_array[i,j], back_avg[jp], back_std[jp]))

        # Precompute the log likelihoods for the sequences matching the 
        # background model, and subtract each kmer individually, to be
        # subbed in with the foreground model scores at each iteration.
        cdef int kmr_len = K*p
        back_logL = np.zeros((N1,nkmers), dtype=np.double)
        cdef double [:,:] back_logL_view = back_logL
        for i in xrange(N1):
            rowsum = 0.
            for j in xrange(N2):
                rowsum += back_s_view[i,j]
            for j in xrange(nkmers):
                jp = j*p
                possum = 0.
                for jj in xrange(kmr_len):
                    possum += back_s_view[i,jj+jp]
                back_logL_view[i,j] = rowsum - possum

        # Perform Expectation Maximization. Use random
        # restarts to compensate for local maxima in the
        # energy landscape.
        restart_vals = []
        cdef double [:] match_avg_view, match_std_view, new_avg_view, new_std_view, denom_view
        cdef double [:,:] M_view, accum_view
        cdef Py_ssize_t [:] II_view
        cdef double match_av, match_sd, logL, max_ref, mmd1, mmd2, val, denom
        cdef Py_ssize_t maxi
        for _ in range(n_init):
            # Initialize a motif.
            if seed_motif is None:
                # Initialize the motif randomly.
                match_motif_avg = np.random.rand(p*k) - 0.5
                match_motif_std = np.zeros([p*k]) + 0.5
            else:
                # Use the motif provided to the function.
                match_motif_avg = seed_motif[0]
                match_motif_std = seed_motif[1]
            match_avg_view = match_motif_avg
            match_std_view = match_motif_std
            
            # Track the likelihoods, so we can determine
            # if we get caught in a loop. Also track the 
            # likelihoods so we can stop if the algorithm
            # has converged.
            likelihoods = []
            lastlogL = None
            lastM = None
            cycle = False
            for __ in xrange(max_iter):
                # Print out a text representation of the current 
                # iteration of the StruM.
                if verbose:
                    print self.text_strum(
                        match_motif_avg, match_motif_std, 
                        avg_range=(-4,4), std_range=(0,2), 
                        colorbar=False)[0]

                ## Do Expectation step once.
                ## Given the motif above, what is the 
                ## probability of each kmer being match?
                M = np.zeros([N1, nkmers], dtype=np.double)
                M_view = M
                with nogil:
                    for i in prange(N1, num_threads=n_threads):
                        for j in xrange(nkmers):
                            logL = 0.
                            jp = j*p
                            for jj in xrange(kmr_len):
                                match_av = match_avg_view[jj]
                                match_sd = match_std_view[jj]
                                logL = logL + log10(norm_p(sequences_data_view[i,jj+jp], match_av, match_sd))
                            M_view[i,j] = logL + back_logL_view[i,j]

                ### Only keep the best alignment for each sequence 
                ### (forward vs. reverse complemented)
                M, II = cleanM(M)

                II = np.array(II, dtype=int)
                M_view = M
                II_view = II

                ### NORMALIZE BY THE ROW USING LOG IDENTITIES
                logL = 0.
                all_denoms = np.zeros((N1/2), dtype=np.double)
                denom_view = all_denoms
                with nogil:
                    for i in prange(N1/2, num_threads=n_threads):
                        maxi = amax(M_view[i, :], nkmers)
                        denom = 0.
                        max_ref = M_view[i,maxi]
                        for j in xrange(nkmers):
                            if maxi != j:
                                denom = denom + doubleten**(M_view[i,j]-max_ref)
                        denom = log10(doubleone+denom)
                        denom = denom + max_ref
                        denom_view[i] = denom
                        for j in xrange(nkmers):
                            M_view[i,j] = doubleten**(M_view[i,j]-denom)
                for i in range(N1/2):
                    logL = logL + denom_view[i]

                ## Check if there is some sort of loop in the
                ## likelihoods. If so, continue until the
                ## maximum is re-reached.
                if logL in likelihoods:
                    if cycle:
                        if logL == cycle_max:
                            print >> err, msg_stop.fromat(__+1)
                            break
                        else:
                            print >> err, msg_cycle
                            for i in range(len(likelihoods)):
                                if likelihoods[i] == logL:
                                    cyc_start = i
                                    cycle = True
                                    break
                            cycle_max = np.max(likelihoods[cyc_start:])

                ## Check for the convergence of the function.
                ## This occurs when the difference between
                ## the log-likelihood from this round is
                ## smaller than some threshold: the
                ## `convergence_criterion`.
                likelihoods.append(logL)
                if lastlogL:
                    if abs(logL - lastlogL) < convergence_criterion:
                        print >> err, msg_cnvrg.format(__+1)
                        break
                lastlogL = logL

                ## M - Step
                ## Use the weights in `M` to learn the mean
                ## and standard deviation from maximimum
                ## likelihood.
                mmd1 = 0.0
                mmd2 = 0.0
                for i in xrange(N1/2):
                    for j in xrange(nkmers):
                        val = M[i,j]
                        mmd1 += val
                        mmd2 += (val*val)
                mmd2 = mmd1 - (mmd2/mmd1)

                new_motif_avg = np.zeros((kmr_len,), dtype=np.double)
                new_motif_std = np.zeros((kmr_len,), dtype=np.double)
                new_avg_view = new_motif_avg
                new_std_view = new_motif_std

                ### Compute the Mean values
                avg_accumulator = np.zeros((N1/2, kmr_len), dtype=np.double)
                accum_view = avg_accumulator
                with nogil:
                    for i in prange(N1/2, num_threads=n_threads):
                        for j in xrange(nkmers):
                            jp = j*p
                            for jj in xrange(kmr_len):
                                accum_view[i,jj] = accum_view[i,jj] + (M_view[i,j]*sequences_data_view[II_view[i], jp+jj])

                for i in xrange(N1/2):
                    for jj in xrange(kmr_len):
                        new_avg_view[jj] = new_avg_view[jj] + accum_view[i,jj]
                for jj in xrange(kmr_len):
                    new_avg_view[jj] = new_avg_view[jj] / mmd1

                ### Compute the Standard Deviations
                avg_accumulator = np.zeros((N1/2, kmr_len), dtype=np.double)
                accum_view = avg_accumulator
                with nogil:
                    for i in xrange(N1/2):
                        for j in xrange(nkmers):
                            jp = j*p
                            for jj in xrange(kmr_len):
                                accum_view[i,jj] = accum_view[i,jj] + (M_view[i,j]*(sequences_data_view[II_view[i], jp+jj]-new_avg_view[jj])**2)
                for i in xrange(N1/2):
                    for jj in xrange(kmr_len):
                        new_std_view[jj] = new_std_view[jj] + accum_view[i,jj]
                for jj in xrange(kmr_len):
                    new_std_view[jj] = new_std_view[jj]/mmd2
                    new_std_view[jj] = sqrt(new_std_view[jj])

                ### Loosen the distributions that have become too specific.
                new_motif_std[new_motif_std < sqrt(lim)] = sqrt(lim)

                ## Store the values for the next iteration.
                match_motif_avg = new_motif_avg[:]
                match_motif_std = new_motif_std[:]
                match_avg_view = match_motif_avg
                match_std_view = match_motif_std
                lastM = M

            # Notify the user if the model never converged.
            if __ == max_iter - 1:
                print >> err, msg_end.format(max_iter)

            # Store the motif found with this initialization
            restart_vals.append(([match_motif_avg, match_motif_std], logL, M, II))

        # Sort the results from random initializations based
        # on their likelihoods.
        restart_vals.sort(key=_sorter_key, reverse=True)
        to_print = []
        for i in xrange(len(restart_vals)):
            to_print.append(restart_vals[i][1])
        print >> err, "Restart Likelihoods:", to_print

        # Use the best motif (based on likelihood) as the
        # StruM
        (match_motif_avg, match_motif_std), logL, M, II = restart_vals[0]
        self.strum = [match_motif_avg, match_motif_std]
        if lim is not None:
            self.strum[1][self.strum[1] < lim] = lim

        # Learn a PWM based on the positions identified
        # during StruM learning.
        pwm_seqs = []
        weights = []
        for i in range(len(M)):
            n = np.argmax(M[i])
            s = sequences_up[II[i]]
            pwm_seqs.append(s[n:n+k+1])
            weights.append(M[i][n])
        self.define_PWM(pwm_seqs, weights=weights)
        self.fit = True

    def update(self, features, func, data=None):
        """Update the StruM to incorporate additional features.

        Using this method will change the behavior of other methods,
        especially the :func:`translate` method is replaced by 
        :func:`func_translate`.

        :param features: Text description or label of the feature(s)
            being added into the model.
        :type features: list of str.
        :param func: The scoring function that produces the additional
            features. These may be computed on sequence alone, or by
            incorporating additional data. The output must be an array
            of shape ``[n, l-1]``, where ``n`` is the number of additional 
            features (``len(features)``) and ``l`` is the length of the 
            sequence being scored. The first argument of the function must
            be the sequence being scored.
        :type func: function.
        :param data: Additional data that is used by the new function. May
            be a lookup table, for example, or a reference to an outside
            file.
        """
        self.features = self.features + features
        self.p += len(features)
        self.func = func
        if data is not None:
            self.func_data = data
        self.translate = self.func_translate
        self.updated = True

    def func_translate(self, seq, **kwargs):
        """Convert sequence from string to structural representation,
            with additional features.

        :param seq: DNA sequence, all uppercase characters,
            composed of letters from set ACGTN.
        :type seq: (str, [args]).
        :param \*\*kwargs: Additional keyword arguments required
            by ``self.func``.

        
        :return: Sequence in structural representation.
        :rtype: 1D numpy array of floats.
        """
        assert (self.updated == True), \
            "Must call ``StruM.update`` first"
        row = []
        for i in range(len(seq[0])-1):
            di = seq[0][i:i+2]
            if 'N' in di:
                row.append(np.zeros([self.p,]))
            else:
                row.append(self.data[:, self.index[di]])
        args = seq[1]
        addition = self.func(seq[0], self.func_data, *args, **kwargs)
        return np.hstack([np.vstack(row), np.vstack(addition).T]).ravel()

    def filter(self,):
        """Update StruM to ignore uninformative position-specific features.

        Features with a high variance, i.e. non-specific features, do not
        contribute to the specificity of the StruM model. Filtering them
        out may increase the signal-to-noise ratio. The position-specific-
        features are rank ordered by their variance, and a univariate 
        spline is fit to the distribution. The point of inflection is 
        used as the threshold for masking less specific features.

        Once this method is run and the attribute `self.filter_mask` is 
        generated, two additional methods will become available:
        :func:`score_seq_filt` and :func:`eval_filt`.
        """
        assert (self.fit == True), \
            "No motif to filter. Must call `train` or `train_EM` first."

        from scipy.interpolate import UnivariateSpline
        idx = np.argsort(self.strum[1])[::-1]
        variance = self.strum[1][idx]
        
        n = len(idx)
        xvals = np.arange(n)

        spl = UnivariateSpline(xvals, variance, s=n/10.)
        d_spl = spl.derivative(1)
        d_ys = d_spl(xvals)

        min_i = np.argmax(d_ys)
        self.var_thresh = spl(min_i)
        self.filter_mask = idx[min_i:]
        self.filtered = True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def score_seq_filt(self, seq, **kwargs):
        """A variation on :func:`score_seq` that masks non-specific features.

        Once the `self.filter_mask` is generated, this method becomes available. 
        This scores a sequence with the precomputed StruM, masking non-specific
        features.

        Refer to :func:`score_seq` for more information about the arguments.
        """
        assert (self.fit == True), \
            "No StruM to score with. Must call `train` or `train_EM` first."
        assert (self.filtered == True), \
            "Must call ``StruM.filter()`` before using ``score_seq_filt``."

        
        cdef double val, val2
        cdef Py_ssize_t i, n_kmers, kmr_len, j, kmer_i, n, jj
        cdef int p = self.p
        cdef int k = self.k
        cdef int n_threads
        cdef double [:] strum_avg = self.strum[0]
        cdef double [:] strum_std = np.zeros(self.strum[1].shape)
        cdef Py_ssize_t [:] filter_mask = self.filter_mask

        n = self.filter_mask.shape[0]

        for i in xrange(self.strum[1].shape[0]):
            strum_std[i] = sqrt(self.strum[1][i])

        strucseq = self.translate(seq, **kwargs)
        cdef double [:] strucseq_view = strucseq
        
        n_kmers = len(seq) - k + 1
        kmr_len = p*(k-1)
        
        by_kmer = np.zeros((n_kmers), dtype=np.double)
        cdef double [:] by_kmer_view = by_kmer

        n_threads = self.n_process
        with nogil:
            for i in prange(n_kmers, num_threads=n_threads):
                kmer_i = i*p
                for jj in xrange(n):
                    j = filter_mask[jj]
                    val = norm_p(strucseq_view[kmer_i+j], strum_avg[j], strum_std[j])
                    val2 = log10(val)
                    by_kmer_view[i] += val2

        return by_kmer

    def plot(self, save_path):
        """Save a graphical representation of the StruM.

        Generates an image displaying the trends of each of the 
        features in the StruM. The line indicates the average 
        (scaled) value for that feature at each position. Shading
        represents +/- 1 standard deviation.

        .. note::
           This generates one row per feature. If you are including 
           many features, your plot may be excessively tall.

        :param save_path: Filename to use when saving the image.
        :type save_path: str.
        """
        logo_vals = np.reshape(self.strum[0], [self.k-1, self.p]).T
        logo_wts  = np.reshape(self.strum[1], [self.k-1, self.p]).T
        new_names = self.features

        n,m = logo_vals.shape
        xs = np.asarray(range(1,m+1))
        colors = ['steelblue']
        figwidth = 3+(m+1)/3.
        figheight = 1+(n)*float(figwidth-3)/m
        
        plt.figure(figsize=[figwidth,figheight])
        override = {
           'verticalalignment'   : 'center',
           'horizontalalignment' : 'right',
           'rotation'            : 'horizontal',
           #'size'                : 22,
           }

        for i in range(n):
            plt.subplot(n,1,i+1)
            up = logo_vals[i] + logo_wts[i]
            dn = logo_vals[i] - logo_wts[i]
            plt.plot(xs, logo_vals[i], color='black', zorder=10)
            y1, y2 = plt.ylim()
            plt.fill_between(xs, up, dn, alpha=0.5, color=colors[i%len(colors)], zorder=1)
            plt.xticks([])
            plt.yticks([])
            plt.xlim([xs[0],xs[-1]])
            plt.ylim([-3,3])
            plt.ylabel(new_names[i], **override)

        plt.xticks(range(1,m+1))
        plt.xlabel("Position")

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.01)
        plt.savefig(save_path, dpi=400)
        plt.close()
    def text_strum(self, avg, std, avg_range=None, std_range=None, 
                   colorbar=False):
        """Gerates a ANSI colored Unicode text representation of a StruM.

        :param avg: The average values of the StruM to plot.
        :type avg: Array of floats.
        :param std: The standard deviation values of the StruM to plot.
        :type std: Array of floats.
        :param avg_range: Eight bins of even width will be generated across
            this range, and the ``avg`` values will be assigned to one of the
            bins.
        :type avg_range: tuple ``(low_value, high_value)``.
        :param std_range: Eight bins of even width will be generated across
            this range, and the ``std`` values will be assigned to one of the
            bins.
        :type std_range: tuple ``(low_value, high_value)``.
        :param colorbar: Whether to include a colorbar
        :type colorbar: bool.
        
        :return: Returns a tuple of strings. The first element is the 
            formatted StruM representation. If ``colorbar=True``, the second
            element is the colorbar string.
        :rtype: tuple of strings.
        """
        # Define things for plotting
        short_ramp = [
            u'\u2581',
            u'\u2582',
            u'\u2583',
            u'\u2584',
            u'\u2585',
            u'\u2586',
            u'\u2587',
            u'\u2588',
        ][::-1]

        black = u'\x1b[1;30;47m{}\x1b[0m'
        white = u'\x1b[1;37;40m{}\x1b[0m'
        yellow = u'\x1b[1;33;40m{}\x1b[0m'
        green = u'\x1b[1;32;40m{}\x1b[0m'
        cyan = u'\x1b[1;36;40m{}\x1b[0m'
        magenta = u'\x1b[1;35;40m{}\x1b[0m'
        blue = u'\x1b[1;34;40m{}\x1b[0m'
        red = u'\x1b[1;31;40m{}\x1b[0m'
        colors = [black,white,yellow,green,cyan,magenta,blue,red,]
        
        # Convert values to bins
        if avg_range is None:
            avg_range = (min(avg), max(avg))
        if std_range is None:
            std_range = (0, max(std))

        avg_binsize = (avg_range[1]-avg_range[0])/(len(colors)) + 0.01*(avg_range[1]-avg_range[0])
        std_binsize = (std_range[1]-std_range[0])/float(len(short_ramp)) + 0.01*(std_range[1]-std_range[0])

        avg_bins = np.asarray((avg - avg_range[0]) // avg_binsize, dtype=int)
        avg_bins[avg_bins >= len(colors)] = len(colors) - 1
        std_bins = np.asarray((std - std_range[0]) // std_binsize, dtype=int)
        std_bins[std_bins >= len(short_ramp)] =len(short_ramp) - 1

        # print avg_binsize
        # print std_binsize

        # Generate formatted string
        strum_str = ""
        for i in range(len(avg_bins)):
            colors[avg_bins[i]]
            short_ramp[std_bins[i]]
            strum_str += colors[avg_bins[i]].format(short_ramp[std_bins[i]])

        if colorbar:
            cbar = u'\x1b[1;30;47m{}\x1b[0m'
            avg_str = u"Avg: {:0.2f}".format(avg_range[0])
            for i in range(len(colors)):
                avg_str += colors[i].format(u'\u2588')
            avg_str += u"{:0.2f}".format(avg_range[1])
            std_str = u"StDev: {:0.2f}".format(std_range[0])
            for i in range(len(colors)):
                std_str += cbar.format(short_ramp[i])
            std_str += u"{:0.2f}".format(std_range[1])
            return (strum_str, avg_str + std_str)
        else:
            return (strum_str,)


cdef inline double normalCDF(double value) nogil:
    """Compute the cumulative density function 
    for the standard normal distribution based
    on a specific case of the ``erfc`` function.

    :param value: The Z-score which provides the upper
        bound of the area to compute.
    :type value: float.
    """
    return half * erfc(-value * M_SQRT1_2);

cdef inline double norm_p(double x, double mu, double sd) nogil:
    """Compute a single tail p-value for a normal distribution.

    :param x: Value to be scored.
    :type x: float.
    :param mu: Arithmetic mean of the distribution ``x`` is being scored 
        against.
    :type mu: float.
    :param sd: Standard deviation of the distribution ``x`` is being scored
        against.
    :type sd: float. 
    """
    cdef double z, p
    z = (x-mu)/sd
    if z > 0:
        z = -z
    p = normalCDF(z)
    return p

cpdef inline str rev_comp(str seq):
    """Reverse complement a DNA sequence.

    :param seq: A DNA sequence.
    :type seq: str.
    """
    cdef Py_ssize_t i, l
    cdef str n, r_seq

    l = len(seq)
    r_seq = ""
    for i in xrange(l):
        n = seq[i]
        r_seq = rev_comp_index[n] + r_seq
    return r_seq

# Only consider the best match on the forward
# or reverse strand, to use use during the
# maximization step.
cdef cleanM(double[:,:] M):
    """Heuristic to choose best orientation of sequences at each iteration of 
    the Expectation Maximization algorithm."""
    cdef Py_ssize_t i, j
    cdef double m1, m2
    cdef list II
    M2 = np.zeros((M.shape[0]/2, M.shape[1]), dtype=np.double)
    cdef double [:,:] M2_view = M2
    II = []
    for i in range(0,len(M),2):
        m1 = np.max(M[i])
        m2 = np.max(M[i+1])
        if m1 > m2: 
            II.append(i)
        else:
            II.append(i+1)
    for i in xrange(len(II)):
        j = II[i]
        M2_view[i] = M[j]
    return M2, II

cdef inline double _sorter_key(val):
    """Equivalent to: ``lambda x:x[1]``"""
    return val[1]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t amax(double[:] array, Py_ssize_t n) nogil:
    """Fast replacement for ``numpy.argmax`` for arrays of known size.

    This function finds the index in an array of the element that has
    the largest value.

    :param array: The array to search.
    :type array: nd.array.
    :param n: The length of ``array``.
    :type n: int.
    """
    cdef Py_ssize_t i, ind
    cdef double m, m2
    m, ind = array[0], 0
    for i in xrange(n):
        m2 = array[i]
        if m2 > m:
            m, ind = m2, i
    return ind