#!/usr/bin/env python

# Imports
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

# Train the motif using the 'groove' related features
motif = strum.StruM(mode='groove')
motif.train(training_sequences, fasta=False)

# Plot the the StruM, and save to the specified filename
motif.plot('FOXA1_strum.png')