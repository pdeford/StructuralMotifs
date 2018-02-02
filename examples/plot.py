#!/usr/bin/env python

from strum import strum

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

motif = strum.StruM(mode='groove')
motif.train(training_sequences, fasta=False)
motif.plot('strumplot.png')