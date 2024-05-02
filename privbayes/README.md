Taken from here:
https://github.com/sdv-dev/SDGym/tree/master/privbayes
and this is the relevent paper:
https://dl.acm.org/doi/pdf/10.1145/3134428

DO NOT USE CONTINIOUS VARIABLES, because the code do binning for continuous variables
See `table.cpp@16`

This looks like another version of the code, perhaps the original before making adaptation to SDGym,
but it could have some difference because its associated paper talks about the NIST 2018 competition,
and not orginal paper from 2017.
https://github.com/journalprivacyconfidentiality/privbayes-nist-jpc/
https://journalprivacyconfidentiality.org/index.php/jpc/article/download/776/723
But this paper does not fit to the code! it is (eps, delta)-DP and does not use in exponential mechanism at all (the code does)!
