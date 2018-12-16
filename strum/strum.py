#!/usr/bin/env python

# MIT License
#
# Copyright (c) 2018 Peter DeFord
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""StruM: Structural Motifs
---------------------------

This package provides functionality for computing structural 
representations of DNA sequence motifs. Estimates for DNA structure
comes from the DiNucleotide Property Database 
(http://diprodb.leibniz-fli.de/).
"""



__version__ = '0.2'
__author__ = 'Peter DeFord'

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
import os
import sys

import numpy as np
from scipy.special import ndtr

###############################################################################
# Line length, 79 characters

def read_diprodb():
	"""Load the values from the DiNucleotide Property Database as a lookup table."""
	data = []
	features = []
	acids = []
	strands = []
	dipro_path = os.path.join(os.path.dirname(__file__), 'data/diprodb_2016.txt')
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

	return np.asarray(data), features, acids, \
		   strands, dinuc_index

class _Pool2(object):
	"""Provide syntax for uniform calling when the number of
	processers being used is 1 (no multithreading).
	"""
	def __init__(self, arg):
		self.arg = arg
	def map(self, func, array):
		out = []
		for thing in array:
			out.append(func(thing))
		return out
	def join(self):
		return
	def close(self):
		return

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
	
	def __init__(self, mode="full", n_process=1, 
				 custom_filter=[], func=None):
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
		:param func: Additional scoring functions to incorporate.
		:type func: function.
		"""
		# Dictate how many processes to run. If n == -1,
		# use all the available cores.
		self.n_process = n_process
		if self.n_process == -1:
			self.n_process = cpu_count()


		# Read the DiProDB data, and filter the matrix based
		# on the provided mode.
		data, feat, acid, strand, di_index = read_diprodb()
		self.index = di_index

		N = data.shape[0]
		masks = {
			"basic"         : [0, 2, 3],
			"groove"        : [6, 7, 8, 9, 10, 11, 12, 13],
			"protein"       : [24, 25, 26, 27, 29, 31],
			"full"          : [i  for i in range(N) if \
							   ((acid[i] == "DNA") or (acid[i] == "B-DNA")) and \
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

		self.data = data[masks[mode], :]
		self.p = self.data.shape[0]
		self.features = []
		for i in masks[mode]:
			self.features.append(feat[i])

		# Normalize the DiProDB data
		mean = np.mean(self.data, axis=1)
		sd = np.std(self.data, axis=1)
		self.data = (self.data - mean.reshape([-1,1])) / sd.reshape([-1,1])

		# self.mins = np.abs(np.min(self.data, axis=1))
		# self.scale = np.hstack([mean, sd]).T

		# Initialize attributes for later use
		self.func = None
		self.func_data = None

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

	def score_seq_filt(self, seq, **kwargs):
		"""A variation on :func:`score_seq` that masks non-specific features.

		Once the `self.filter_mask` is generated, this method becomes available. 
		This scores a sequence with the precomputed StruM, masking non-specific
		features.

		Refer to :func:`score_seq` for more information about the arguments.
		"""
		try:
			self.filter_mask
		except:
			raise ValueError('`self.filter_mask` not specified. Call `StruM.filter()` first!')
		strucseq = self.translate(seq, **kwargs)
		n = len(strucseq)
		kmr_len = self.strum.p*(self.strum.k - 1)
		kmer_stack = np.vstack([
			strucseq[i:i + kmr_len] \
			for i in range(0, n - kmr_len + self.strum.p, self.strum.p)
			])[:, self.filter_mask]
		by_pos = self.norm_p(
			kmer_stack, self.strum[0][self.filter_mask], self.strum[1][self.filter_mask]**2)
		by_pos = np.log10(by_pos)
		by_kmer = np.sum(by_pos, axis=1)
		return by_kmer

	def eval_filt(self, struc_kmer):
		""" A variation on :func:`eval` that masks non-specific features.

		Once the `self.filter_mask` is generated, this method becomes available. 
		This compares the structural representation of a sequence to the 
		StruM.

		Refer to :func:`eval` for more information about the arguments.
		"""
		try:
			self.filter_mask
		except:
			raise ValueError('`self.filter_mask` not specified. Call `StruM.filter()` first!')
		return np.sum(np.log10(10.**-300 + self.norm_p(
			struc_kmer[self.filter_mask], self.strum[0][self.filter_mask], self.strum[1][self.filter_mask]**2)))
	

	def rev_comp(self, seq):
		"""Reverse complement (uppercase) DNA sequence.

		:param seq: DNA sequence, all uppercase characters,
			composed of letters from set ACGTN.
		:type seq: str.
		
		:return: Reverse complement of ``seq``.
		:rtype: str.
		"""
		nucs = "ACGT"
		index = dict(zip(nucs, nucs[::-1]))
		index['N'] = 'N'
		return "".join([index[n] for n in seq][::-1])

	def translate(self, seq, **kwargs):
		"""Convert sequence from string to structural representation.

		:param seq: DNA sequence, all uppercase characters,
			composed of letters from set ACGTN.
		:type seq: str.
		:param \*\*kwargs: Ignored
		
		:return: Sequence in structural representation.
		:rtype: 1D numpy array of floats.
		"""
		row = []
		for i in range(len(seq)-1):
			di = seq[i:i+2]
			if 'N' in di:
				row.append(np.zeros([self.p,]))
			else:
				row.append(self.data[:, self.index[di]])
		return np.vstack(row).ravel()
		# For v_00 v_10 ... v_k0 v_01 ... v_pk

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
		# For v_00 v_10 ... v_k0 v_01 ... v_pk

	def train(self, training_sequences, 
			  weights=None, lim=None, **kwargs):
		"""Learn structural motif from a set of known binding sites.

		:param training_sequences: Training set, composed of gapless 
			alignment of binding sites of equal length.
		:type training_sequences: list of str. Or if updated, list of tuples: (str, [args])
		:param weights: Weights to associate with each of the sequences
			in ``training_sequences`` to use in learning the motif.
		:type weights: 1D array of floats.
		:param lim: Minimum value allowed for variation in a given
			position-specific-feature. Useful to prevent *any*
			deviation at that position from resulting in a probability of
			0.
		:type lim: float
		
		:return: None. Defines the structural motif ``self.strum`` and the
			corresponding position weight matrix ``self.PWM``.
		"""
		data = []
		for example in training_sequences:
			data.append(self.translate(example, **kwargs))

		arr = np.asarray(data)
		if weights is None:
			weights = np.ones(arr.shape[0])
		average = np.average(arr, axis=0, weights=weights)
		variance = np.average((arr-average)**2, axis=0, weights=weights)
		self.strum = [average, np.sqrt(variance)]
		if lim is not None:
			self.strum[1][self.strum[1] < lim] = lim
		if self.func is not None:
			training_sequences = [x[0] for x in training_sequences]
		self.k = len(training_sequences[0])
		self.define_PWM(training_sequences, weights=weights)
		


	def calc_z(self, x, mu, var):
		"""Calculate Z-scores for values based on mean and standard deviation.

		:param x: Value or values of interest.
		:type x: float, numpy array.
		:param mu: Average of population from which ``x`` was sampled.
		:type mu: float, numpy array (``x.shape[1]``).
		:param var: Variance of population from which ``x`` was sampled.
		:type var: float, numpy array (``x.shape[1]``).
		
		:return: The Z-score for all values in ``x``.
		:rtype: ``type(x)``
		"""
		z = -np.absolute(x-mu)/np.sqrt(var)
		return z

	def norm_p(self, x, mu, var):
		"""Finds the one-tail p-value for values from the standard normal 
		distribution. Adds a 'pseudocount' of 10e-300 to avoid underflow.

		:param x: Value or values of interest.
		:type x: float, numpy array.
		:param mu: Average of population from which ``x`` was sampled.
		:type mu: float, numpy array (``x.shape[1]``).
		:param var: Variance of population from which ``x`` was sampled.
		:type var: float, numpy array (``x.shape[1]``).
		
		:return: The p-value for all values in ``x`` relative to ``mu``
			and ``var``.
		:rtype: ``type(x)``
		"""
		z = self.calc_z(x, mu, var)
		ps = ndtr(z)
		# m = ps > 0.5
		# ps[m] = 1 - ps[m]
		return ps + 10.**-300

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

	def train_EM(self, data, fasta=True, params=None, k=10,
		max_iter=1000, convergence_criterion=0.001, 
		random_seed=0, n_init=1, lim=None, seqlength=None,
		background=None, seed_motif=None):

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
			dinucleotide. E.g. ``AA\t0.0625\nAC\t0.0625\n...``
		:type background: str.
		:param seed_motif: Optional. A `StruM.strum` to use for 
			initializing the Expectation-Maximization algorithm.
			If set, ``k`` will be replaced by the corresponding
			value from the ``seed_motif``'s shape. ``n_init`` will
			also be reset to 1.
		:type seed_motif: ``StruM.strum``.

		:return: None. Defines the structural motif ``self.strum`` and the
			corresponding position weight matrix ``self.PWM``.
		"""

		err = sys.stderr

		if seed_motif is not None:
			k = seed_motif[0] / self.p
			n_init = 1

		self.k = k + 1
		K = k
		p = self.p

		# Dictate the number of cores to use
		if self.n_process == 1:
			Pool = _Pool2

		# Set the random seed for reproducibility
		if random_seed is not 0:
			np.random.seed(random_seed)

		# Read in the data and define the sequences
		if fasta:
			headers, sequences = self.read_FASTA(data)
		else:
			sequences = data
		sequences_up = []
		for i in range(len(sequences)):
			sequences_up.append(sequences[i].upper())
			sequences_up.append(self.rev_comp(sequences[i].upper()))

		# Ensure that all of the sequences are the same 
		##length. If no seqlength is passed, use the length
		##of the shortest sequence. Otherwise, throw out
		##sequences shorter than the required length. Select
		##the subsequence of the appropriate length from the
		##center of the original sequence.
		# Convert to structural space at the same time.
		if seqlength is None:
			seqlength = min(len(s) for s in sequences_up)

		sequences_data = []

		for s in sequences_up:
			l = len(s)
			if l >= seqlength:
				i = (l - seqlength)//2
				s = s[i:i+seqlength] 
				sequences_data.append( self.translate(s) )

		sequences_data = np.vstack(sequences_data)
		nseqs = sequences_data.shape[0]
		nkmers = seqlength - k
		print >> err, "Retaining {} out of {} sequences, based on length (>{}bp)".format(nseqs/2, len(sequences), seqlength)

		# Initialize the background motif. Since the matrix
		# of data has been centered and scaled, we expect a
		# fully random distribution to result in a mean of
		# 0, and a variance of 1.
		back_motif = [0., 1.]

		# Precompute the probabilities for each position
		# matching to the background.
		# big_scale = 10.**(-300./(k*p))
		exp_scale = 1./((seqlength-k)*p)
		if background is None:
			# Assume equal representation of each dinucleotide.
			# Results in ~N(0, 1) because that was how the
			# data was normalized
			back_s = self.norm_p(sequences_data, 0., 1.) #**exp_scale #* big_scale
		elif background is 'compute':
			# Use the training sequences to compute the background
			# model. Allow every position in every sequence to
			# contribute.
			all_pos = np.reshape(sequences_data, (-1, p))
			back_avg = np.average(all_pos, axis=0)
			back_std = np.average(all_pos, axis=0)
			back_s = self.norm_p(all_pos, back_avg, back_std**2)
			back_s.reshape(sequences_data.shape)
		else:
			# Read in a precomputed background model. This should
			# be a tab delimited file with two columns -- uppercase
			# dinucleotide followed by its representation in the 
			# organism. E.g.
			# AA	0.0625
			# AC	0.0600
			# AG	0.0650
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
					di_vals.append(di_val)
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
			all_pos = np.reshape(sequences_data, (-1, p))
			back_s = self.norm_p(all_pos, back_avg, back_std**2)
			back_s.reshape(sequences_data.shape)


		# scaler = 10.**(-300./(k*p*seqlength))
		n = back_s.shape[1]
		x = np.product(back_s ** (1./n) , axis=1)
		c = (10.**-5)**(1./n)/x
		big_scale = c.reshape([-1,1])
		back_s *= big_scale
		back_L = np.product(back_s[:, k*p:], axis=1)

		# Only consider the best match on the forward
		# or reverse strand, to use use during the
		# maximization step.
		def cleanM(M):
			II = []
			for i in range(0,len(M),2):
				m1 = np.max(M[i])
				m2 = np.max(M[i+1])
				if m1 > m2: 
					II.append(i)
				else:
					II.append(i+1)
			return M[II], II

		# Perform Expectation Maximization. Use random
		# restarts to compensate for local maxima in the
		# energy landscape.
		restart_vals = []
		for i in range(n_init):
			if seed_motif is None:
				# Initialize motif, randomly
				match_motif = [np.random.rand(p*k) - 0.5, 
							   np.zeros([p*k]) + 0.5]
			else:
				match_motif = seed_motif

			# Track the likelihoods, so we can determine
			# if we get caught in a loop. Also track the 
			# likelihoods so we can stop if the algorithm
			# has converged.
			likelihoods = []
			lastlogL = None
			lastM = None
			cycle = False

			for __ in range(max_iter):
				##Do Expectation step once.
				##Given the motif above, what is the 
				##probability of each kmer being match?
				M = np.zeros([nseqs, nkmers])
				L_upto = np.ones([nseqs], 
					dtype=np.float64)
				L_after = back_L.copy()

				# print match_motif[0]
				# print match_motif[1]

				for i in range(0, (seqlength-k)*p, p):
					kmer_stack = sequences_data[:, i:i+k*p]
					try:
						L_stack = np.product(self.norm_p(
							kmer_stack, match_motif[0], 
							match_motif[1]**2)*big_scale,
							axis=1)
					except:
						print >> err, match_motif[0]
						print >> err, match_motif[1]
						print >> err, "YOU BROKE IT"
						print >> err, __, i, seqlength
						L_stack = np.product(self.norm_p(
							kmer_stack, match_motif[0], 
							match_motif[1]**2)*big_scale,
							axis=1)
						quit()

					M[:,i//p] = L_upto * L_stack * L_after

					back_stack = back_s[:, i:i+k*p]
					# print back_stack[0][-p:]
					change_up = np.product(
									back_stack[:, :p], axis=1)
					change_down = np.product(
									back_stack[:, (k-1)*p:], axis=1)
					L_upto *= change_up
					L_after /= change_down
					
					# print L_upto[0], L_stack[0], L_after[0], change_up[0], change_down[0]

				#| Instead of working in log space, I think I should
				#| multiply all of the probabilities by a very large 
				#| number (like 10.**300).
				#| The probabilities are bounded by:	0 <  p  < 1
				#| .`. the scaled probs are bounded by: 0 < p*k < k
				#| So instead of working in log space, and starting 
				#| my accumulator at 0, and instead of normal space
				#| and starting my accumulator at 1., I can use
				#| 'normal scaled' space, and start my accumulator
				#| at `k`. Since all of these values get normalized
				#| to the values of the row anyway, this should still
				#| largely avoid overflow as much as possible, and
				#| maintain the ratios between values in the row.
				#| 
				#| This has the benefits of:
				#|     a) Removing all of the log operations.
				#|     b) Removing the necessity for sorting and
				#|        using weird log identities to norm the
				#|        row.
				#| Together this should speed up this part a lot.
				#|
				#| If I do this, the normalization section will look
				#| like the following:

				M, II = cleanM(M)

				rowsum = np.sum(M, axis=1) + 10.**-300
				M /= rowsum.reshape([-1,1])

				logL = np.sum(np.log(rowsum))

				#| BOOM.
				#|
				#| Looks great. One potential problem:
				#| Whereas before I had a sum of things in log space,
				#| in normal space I am working with products. If I 
				#| factor out the scaling factor, `k`, it goes 
				#| something like this:
				#|     p   = product(v_i for i=0 to n)
				#|     p_k = product(k*v_i for i=0 to n)
				#|         = k^n * product(v_i for i=0 to n)
				#|         = k^n * p
				#| Hopefully `p` is a small enough number to counter-
				#| act the immensity of the number that would be `k^n`.
				#| I'm not sure that will necessarily be the case 
				#| though. I may need to think more about how to 
				#| handle this problem. 
				#|
				#| I could dynamically manage `k` based on the size of
				#| the values I encounter, but that would require
				#| checking `k` and would introduce more computations
				#| which is what I was trying to avoid. Maybe I will
				#| need to try some things and see what `k` works well.
				#| Then I could make that the default, and let users 
				#| adjust _only if/as necessary_.
				#|
				#| Maybe a place to start would be to divide the
				#| the largest float by the length of the sequences
				#| being considered times the number of features, or
				#| do some nth root...

				# Check if there is some sort of loop in the
				# likelihoods. If so, continue until the
				# maximum is re-reached.
				if logL in likelihoods:
					if cycle:
						if logL == cycle_max:
							print >> err, msg_stop.format(__+1)
							break
					else:
						print >> err, msg_cycle
						for i,l in enumerate(likelihoods):
							if l == logL:
								cyc_start = i
								cycle = True
						cycle_max = np.max(likelihoods[cyc_start:])
						
				# Check for the convergence of the function.
				# This occurs when the difference between
				# the log-likelihood from this round is
				# smaller than some threshold: the
				# `convergence_criterion`.
				likelihoods.append(logL)
				if lastlogL:
					if abs(logL - lastlogL) < \
						convergence_criterion:
						print >> err, msg_cnvrg.format(__+1)
						break
				lastlogL = logL

				# M - Step
				## Use the weights in `M` to learn the mean
				## and standard deviation from maximimum
				## likelihood.
				filtered_seq_data = sequences_data[II]

				match_motif_denom = [
					np.sum(np.ravel(M)),
					np.sum(np.square(np.ravel(M)))
					]
				mmd = match_motif_denom
				mmd[1] = mmd[0] - (mmd[1]/mmd[0])


				## The mean
				adjustment = np.zeros([nseqs/2, K*p], 
					dtype=np.float64)
				for i in range(0, (seqlength - k)*p, p):
					kmer_stack = filtered_seq_data[:, i:i+k*p]
					col = i//p
					multiplier = M[:, col:col+1] / mmd[0]
					adjustment += kmer_stack*multiplier
				max_motif = np.sum(adjustment, axis=0)

				## The standard deviation
				adjustment = np.zeros([nseqs/2, K*p], 
					dtype=np.float64)
				for i in range(0, (seqlength - k)*p, p):
					kmer_stack = filtered_seq_data[:, i:i+k*p]
					col = i//p
					multiplier = M[:, col:col+1] * \
						((kmer_stack - max_motif)**2) / mmd[1]
					adjustment += multiplier
				motif_error = np.sum(adjustment, axis=0)

				thresh = 0.001 #| hardcoded for now
				motif_error[motif_error < thresh] = thresh

				## Store these values
				match_motif = [max_motif, motif_error]
				lastM = M

			if __ == max_iter - 1:
				print >> err, msg_end.format(__ + 1)

			# Store the motif found with this initialization
			restart_vals.append((match_motif, logL, M, II))

		# Sort the results from random initializations based
		# on their likelihoods.
		## Use an actual function instead of lambda because
		## Cython didn't like it?
		restart_vals.sort(key=_sorter_key, reverse=True)
		to_print = []
		for i in range(len(restart_vals)):
			to_print.append(restart_vals[i][1])
		print >> err, "Restart Likelihoods:", to_print

		# Use the best motif (based on likelihood) as the
		# StruM
		match_motif, logL, M, II = restart_vals[0]
		self.strum = match_motif
		if lim is not None:
			self.strum[1][self.strum[1] < lim] = lim

		# Learn a PWM based on the positions identified
		# during StruM learning.
		pwm_seqs = []
		for i in range(len(M)):
			n = np.argmax(M[i])
			s = sequences_up[II[i]]
			pwm_seqs.append(s[n:n+k+1])

		self.define_PWM(pwm_seqs)

	def score_seq(self, seq, **kwargs):
		"""Scores a sequence using pre-calculated StruM.

		:param seq: DNA sequence, all uppercase characters,
			composed of letters from set ACGTN.
		:type seq: str.

		:return: Vector of scores for similarity of each kmer
			in ``seq`` to the StruM.
		:rtype: 1D array.
		"""
		strucseq = self.translate(seq, **kwargs)
		n = len(strucseq)
		kmr_len = self.p*(self.k - 1)
		kmer_stack = np.vstack([
			strucseq[i:i + kmr_len] \
			for i in range(0, n - kmr_len + self.p, self.p)
			])
		by_pos = self.norm_p(
			kmer_stack, self.strum[0], self.strum[1]**2)
		by_pos = np.log10(by_pos)
		by_kmer = np.sum(by_pos, axis=1)
		return by_kmer

	def eval(self, struc_kmer):
		"""Compares the structural representation of a sequence 
		to the StruM.

		:param struc_kmer: A kmer that has been translated to 
			structure-space via :func:`translate`.
		:type struc_kmer: output of :func:`translate`.
		
		:return: *log* score of similarity of kmer to StruM.
		:rtype: float.
		"""
		return np.sum(np.log10(10.**-300 + self.norm_p(
			struc_kmer, self.strum[0], self.strum[1]**2)))

	def define_PWM(self, seqs, weights=None):
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
		nuc_index = dict(zip("ACGT", range(4)))
		if weights is None:
			weights = [1.0] * len(seqs)
		pwm = np.zeros([4,self.k])
		for i, seq in enumerate(seqs):
			for j, n in enumerate(seq):
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

	def plot(self, save_path):
		"""Save a graphical representation of the StruM.

		Generates an image displaying the trends of each of the 
		features in the StruM. The line indicates the average 
		(scaled) value for that feature at each position. Shading
		represents +/- 1 standard devation.

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

def _sorter_key(val):
	return val[1]