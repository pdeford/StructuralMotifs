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



__version__ = '0.1'
__author__ = 'Peter DeFord'

import os
from multiprocessing import cpu_count, Pool
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

class FastStruM(object):
	"""FastStruM: Learn Strucural Motif, quickly
	==================================================
	Differs from the standard StruM in that it cannot
	incorporate additional features, it requires all 
	sequences to have the same length (or else will 
	filter to the shortest length), and uses heuristics
	to speed up scoring.
	"""
	
	def __init__(self, mode="full", n_process=1, 
				 custom_filter=[], func=None):
		"""Create a FastStruM object.

		:param mode: Defines which subset of available 
			features in the DiProDB table to use. Choose from: 
			['basic', 'protein', 'groove', 'proteingroove', 'unique', 
			 'full', 'nucs', 'custom']
			* basic -- Twist, Rise, Bend.
			* protein -- (for DNA-protein complex) Roll, Twist, 
			  Tilt, Slide, Shift, Rise.
			* groove -- Major Groove Width, Major Groove Depth, 
			  Major Groove Size, Major Groove Distance, Minor 
			  Groove Width, Minor Groove Depth, Minor Groove Size, 
			  Minor Groove Distance.
			* proteingroove -- The union of the "protein" and 
			  "groove" filters.
			* unique -- Filters the table for the first occurrence
			  of each type of feature.
			* full -- All available DNA features.
			* nucs -- Adenine content, Guanine content, Cytosine 
			  content, Thymine content.
			* custom -- Manually select desired features.
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
		self.features = feat
		self.index = di_index

		N = data.shape[0]
		masks = {
			"basic"         : [0, 2, 3],
			"groove"        : [6, 7, 8, 9, 10, 11, 12, 13],
			"protein"       : [24, 25, 26, 27, 29, 31],
			"full"          : [i  for i in range(N) if \
							   (acid[i] == "DNA") and \
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

		# Normalize the DiProDB data
		mean = np.mean(self.data, axis=1)
		sd = np.std(self.data, axis=1)
		self.data = (self.data - mean.reshape([-1,1])) / sd.reshape([-1,1])

		# self.mins = np.abs(np.min(self.data, axis=1))
		# self.scale = np.hstack([mean, sd]).T

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

	def translate(self, seq):
		"""Convert sequence from string to structural representation.

		:param seq: DNA sequence, all uppercase characters,
			composed of letters from set ACGTN.
		:type seq: str.
		
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

	def train(self, training_sequences, 
			  weights=None, lim=None):
		"""Learn structural motif from a set of known binding sites.

		:param training_sequences: Training set, composed of gapless 
			alignment of binding sites of equal length.
		:type training_sequences: list of str.
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
		self.k = len(training_sequences[0])
		for example in training_sequences:
			data.append(self.translate(example))

		arr = np.asarray(data)
		if weights is None:
			weights = np.ones(arr.shape[0])
		average = np.average(arr, axis=0, weights=weights)
		variance = np.average((arr-average)**2, axis=0, weights=weights)
		self.strum = [average, np.sqrt(variance)]
		if lim is not None:
			self.strum[1][self.strum[1] < lim] = lim
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
		z = (x-mu)/np.sqrt(var)
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
		m = ps > 0.5
		ps[m] = 1 - ps[m]
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
		random_seed=0, n_init=1, lim=None, seqlength=None):

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

		:return: None. Defines the structural motif ``self.strum`` and the
			corresponding position weight matrix ``self.PWM``.
		"""

		err = sys.stderr

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
		back_s = self.norm_p(sequences_data, 0., 1.) #**exp_scale #* big_scale
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
			# Initialize motif, randomly
			match_motif = [np.random.rand(p*k) - 0.5, 
						   np.zeros([p*k]) + 0.5]

			# Track the likelihoods, so we can determine
			# if we get caught in a loop. Also track the 
			# likelihoods so we can stop if the algorithm
			# has converged.
			likelihoods = []
			lastlogL = None
			latM = None
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
							match_motif[1])*big_scale,
							axis=1)
					except:
						print >> err, match_motif[0]
						print >> err, match_motif[1]
						print >> err, "YOU BROKE IT"
						print >> err, __, i, seqlength
						L_stack = np.product(self.norm_p(
							kmer_stack, match_motif[0], 
							match_motif[1])*big_scale,
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
			s = sequences_up[i]
			pwm_seqs.append(s[n:n+k+1])

		self.define_PWM(pwm_seqs)

	def score_seq(self, seq):
		"""Scores a sequence using pre-calculated StruM.

		:param seq: DNA sequence, all uppercase characters,
			composed of letters from set ACGTN.
		:type seq: str.

		:return: Vector of scores for similarity of each kmer
			in ``seq`` to the StruM.
		:rtype: 1D array.
		"""
		strucseq = self.translate(seq.upper())
		n = len(strucseq)
		kmr_len = self.p*(self.k - 1)
		kmer_stack = np.vstack([
			strucseq[i:i + kmr_len] \
			for i in range(0, n - kmr_len + self.p, self.p)
			])
		by_pos = self.norm_p(
			kmer_stack, self.strum[0], self.strum[1])
		by_pos = np.log(by_pos)
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
		return np.sum(np.log(10.**-300 + self.norm_p(
			struc_kmer, self.strum[0], self.strum[1])))

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

def _sorter_key(val):
	return val[1]