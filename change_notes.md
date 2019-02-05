# DIFFERENCES FROM (StruM) LEGACY MODEL

1. Uses the standard deviation instead of the variance 
   for computing the Z-score. (Mistake in old version).
2. For EM, trims all sequences to the same length.
3. For EM, random initialization of background motif, uses
   random values from [-0.5, 0.5] instead of [0, 1] for mu
   values.
4. Avoid working in log space for likelihoods.
   a. Don't log things initially
   b. Replace lines 478-489 (old) with lines (602-605)
   c.
5. Hardcoded threshold is 0.001 (new) instead of 0.0001 
   (old). Should match `lim` if specified.
6. Provide option to filter out low specificity (high variance)
   features.

# BUG FIXES
1. Actually pass variance to `norm_p`. Was just passing 
   standard deviation in all cases.

# DIFFERENCES FROM (strum_v2) LEGACY MODEL
1. Roll back change (4) above. Using log space again.
2. Have `thresh` be `lim` if specified.
3. Convert from Python to Cython to optimize speed.
4. Include a text representation of a StruM.
    - Can be turned on for each iteration of train_EM if 
      desired.
5. Switch p-value source from `scipy.stats.ndtr` to a specific
   case of the C function `erfc` from `match.h`.
    - https://stackoverflow.com/a/18786808