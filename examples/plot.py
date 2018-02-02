#!/usr/bin/env python

import random

from strum import strum

random.seed(141)

sequences = ["".join([random.choice("ACGT") for i in range(10)]) for i in range(10)]

motif = strum.StruM(mode='groove')
motif.train(sequences, fasta=False)
motif.plot('strumplot.png')