#!/usr/bin/env python

# Imports
import numpy as np
from strum import strum

# 50 example sequences that contain the sequence GATTACA
training_sequences = [
    "CGATTACAGATCTCCCGCGACCCTT", "GATGATTACAAGATGCGTCGAATAT",
    "AGCCCTGTCCGCAGATTACAACCAC", "AGTTAACTCCCTAGATTACATTTGT",
    "CGCTACAAAGTAAAGGAGATTACAT", "CTGATTACATCCTGTCCGAGGCGTG",
    "CGATTACATTCAACTAGATGCGCGC", "GAACGGCATGGGCGATTACAACACT",
    "CGGGGTGATCGTAATGTGATTACAA", "TGGTGCCCTGATTACACCTTACATG",
    "GGCGAAGATTACATCCTCGGCCCAT", "ATCGGATTACATGTACTCGTCCACG",
    "TCGGGGATTACAGGGGAGACGCTTA", "AGGTAGATTACATCGTTTTATTAGT",
    "TATTGTGCTCGATTACAAGCAGGCC", "CAGACCGCTTACACGTTGATTACAA",
    "GACACCCTCGATTACACCTCGTATA", "GGAACCGCGCGGATTACACGCGAGA",
    "GTTGATTACAAGGGAAACATACTTG", "CCCACACATTAGCTCGAGATTACAT",
    "GCAGAGTACCCTGCGGCGATTACAA", "ATACTCACGCATACAGATTACAAGA",
    "GGTATGCATCGCGATTACAGCACTG", "GGATTACAGTGAGCCTGCACCTTGA",
    "TTGGATTACATGGCCAAACTCCACT", "GGCCGGCAGAGATTACACTAGAGAG",
    "ACAGATTACAGTGCAAATTGAGCAG", "CCACTGCATGACTGGATTACAGGCA",
    "CATGCCGGCGGTTAACGGATTACAC", "CCGATTACAAGTGCTCTGCACGGCG",
    "CATATAGAGGCGATTACAGCGTATC", "GGAACGATTACAGTGAGACTGCTCC",
    "ATGATTACAGCGAAACGTATTCAAA", "TTTTCGGATGATTACACATTCTTCT",
    "GTACAATGCATCGCGATTACAACAC", "GGATTACAAGTATCTGCCTGGATAC",
    "CTCCCGATTACATCAGGTACGTCCT", "TAGAGAAGATTACAGCCTACTATTG",
    "AAGCTTTGGGCCGTACGATTACATC", "GTAAGATTACAAGTTCAGGGTGATC",
    "CATGATTACATTGGCGCCGACCTAC", "GCTGGATTACAATCATACCCGTGTA",
    "GGTTAGGGATTACAAACAAGACGTG", "GACCGAGGTCTGATTACACTCCATC",
    "ATAGACGCGATTACAAGCACTCTAA", "TTTCCGTTCTGCAGCTGATTACAAC",
    "GGATTACACGCCTTCTCAAGCAGTG", "ATCCTAACAGGATTACAAGAATTAC",
    "TATGAAGCTGAAGAAGATTACAGCA", "CCTGTCTCAGATTACAGCACGGCGG",
]

# Initialize a new StruM object, using the DNA groove related 
# features from the DiProDB table. Specify to use 4 cpus
# when doing EM.
motif = strum.StruM(mode='groove', n_process=4)

# Train the model on the training sequences using expectation
# maximization, ensuring that that the variation of all 
# position-specific features is at least 10e-5 (lim). Use
# a random_seed for reproducibility, and repeat with 20
# random restarts.
k = 8
motif.train_EM(training_sequences, fasta=False, k=k, 
    lim=10**-5, random_seed=620, n_init=20)

# Examine the output, by identifying the best matching
# kmer in each of the training sequences.
out = []

for sequence in training_sequences:
    rseq = motif.rev_comp(sequence)
    s1 = motif.score_seq(sequence)
    s2 = motif.score_seq(rseq)
    i1 = np.argmax(s1)
    i2 = np.argmax(s2)
    if s1[i1] > s2[i2]:
        seq = sequence
        i = i1
        s = "+"
    else:
        seq = rseq
        i = i2
        s = "-"
    out.append(seq[i:i+k])
    print "{}{: <2} {: >{}} {} {}".format(
        s, i, seq[:i].lower(), len(seq)-k, 
        seq[i:i+k].upper(), seq[i+k:].lower()
        )

# Summarize these best matches with a simple PWM.
nucs = dict(zip("ACGT", range(4)))
PWM = np.zeros([4,k])
for thing in out:
    for i,n in enumerate(thing):
        PWM[nucs[n], i] += 1

PWM /= np.sum(PWM, axis=0)
for row in PWM:
    print row
