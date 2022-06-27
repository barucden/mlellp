import sys

if sys.version_info >= (3, 8):
    import math
    def _binom(n, k):
        return math.comb(n, k)
else:
    def _binom(n, k):
        if not 0 <= k <= n:
            return 0
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok

def comb(n, k):
    return _binom(n, k)
