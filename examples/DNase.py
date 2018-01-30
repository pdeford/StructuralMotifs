#!/usr/bin/env python

# Imports
import numpy as np
import sys

import bx.bbi.bigwig_file
import StruM

# Specify the path to where you downloaded the bigwig
# file. E.g. "/Users/user/Downloads/ENCFF111KJD.bigWig"
DNase_bigwig_path = sys.argv[1]

# Define the function to be used by the StruM when
# converting from sequence-space to structural-space.
# NOTE: This function takes additional parameters.
def lookup_DNase(data, chrom, start, end):
	"""Lookup the signal in a bigWig file, convert NaNs to
	0s, ensure strandedness, and return the modified signal.

	Parameters:
		data : (str) - Path to the bigWig file.
		chrom : (str) - Chromosome name.
		start : (int) - Base pair position on the chromsome of
						the beginning of the region of interest.
		end : (int) - Base pair position on the chromsome of
					  the end of the region of interest.

	Returns:
		trace : (1D array) - The signal extracted from the
							 bigWig file for the region of
							 interest.
	"""
	# Open file
	bwh = bx.bbi.bigwig_file.BigWigFile(open(data))
	# Lookup signal for regions
	trace = bwh.get_as_array(chrom, min(start,end), max(start, end)-1)
	# Clean up NaNs
	trace[np.isnan(trace)] = 0.0
	# Ensure strandedness
	if start > end:
		trace = trace[::-1]
	return trace

# Some example sequences and their chromosome positions, 
# from human build hg19.
training_data = [
	['GAGATCCTGGGTTCGAATCCCAGC', ('chr6', 26533165, 26533141)],
	['GAGATCCTGGGTTCGAATCCCAGC', ('chr19', 33667901, 33667925)],
	['GAGGTCCCGGGTTCGATCCCCAGC', ('chr6', 28626033, 28626009)],
	['GAGGTCCTGGGTTCGATCCCCAGT', ('chr6', 28763754, 28763730)],
	['GGGGGCGTGGGTTCGAATCCCACC', ('chr16', 22308468, 22308492)],
	['GGGGGCGTGGGTTCGAATCCCACC', ('chr5', 180614647, 180614671)],
	['AAGGTCCTGGGTTCGAGCCCCAGT', ('chr11', 59318028, 59318004)],
	['GAGGTCCCGGGTTCAAATCCCGGA', ('chr1', 167684001, 167684025)],
	['GAGGTCCCGGGTTCAAATCCCGGA', ('chr7', 128423440, 128423464)],
]

# Initialize a new StruM object.
motif = StruM.StruM()

# Update the StruM to incorporate the function
# defined above, drawing on the bigWig as the 
# data source.
motif.update(data=DNase_bigwig_path, func=lookup_DNase, 
	features=['k562_DNase'])

# Train the model using the modified StruM and the example 
# data above
motif.train(training_data)

# Evaluate the similarity to the model of a new sequence.
seq = 'GAGGTCCCGGGTTCAATCCCCGGC'
position = ['chr2', 157257305, 157257329]

rseq = motif.rev_comp(seq)
rposition = [position[0], position[2], position[1]]

s = [
	motif.score_seq(seq, *position),
	motif.score_seq(rseq, *rposition)
]

strands = ['-', '+']
print "Best match found on the '{}' strand".format(strands[np.argmax(s)])