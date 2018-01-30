#!/usr/bin/env python

# Imports
import numpy as np
from strum import strum

# Sequences representing some of the variability of the 
# FOXA1 binding site.
training_sequences = [
    "CTGTGCAAACA",
    "CTAAGCAAACA",
    "CTGTGCAAACA",
    "CTGTGCAAACA",
    "CAGAGCAAACA",
    "CTAAGCAAACA",
    "CTGTGCAAACA",
    "CAATGTAAACA",
    "CTGAGTAAATA",
]

# Initialize a new StruM object, using the basic features
# from the DiProDB table.
motif = strum.FastStruM(mode='basic')

# Use the training sequences to define the StruM,
# ensuring that that the variation of all position-specific
# features is at least 10e-5 (lim)
motif.train(training_sequences, lim=10**-5)

# Print out what the PWM would look like for these sequences
motif.print_PWM(True)

# Define an example sequence to analyze.
test_sequence = "ACGTACTGCAGAGCAAACAACTGATCGGATC"
# Reverse complement it, as the best match may not be on
# the forward strand.
reversed_test = motif.rev_comp(test_sequence)

# Get a score of the similarity for each kmer in the test 
# sequence to the StruM.
forward_scores = motif.score_seq(test_sequence)
reverse_scores = motif.score_seq(reversed_test)

# Find the best match.
## Determine the strand the best match lies on
if np.max(reverse_scores) > np.max(forward_scores):
    score, seq = reverse_scores, reversed_test
    strand = "-"
else:
    score, seq = forward_scores, test_sequence
    strand = "+"
## Find position in the sequence
best_pos = np.argmax(score)
best_seq = seq[best_pos : best_pos + motif.k]
best_score = score[best_pos]

# Print the results to screen
print "\n", strand, best_pos, best_seq, best_score