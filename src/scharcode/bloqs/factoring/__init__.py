"""Subroutines of the approximate modular exponentiation oracle (Algorithm `alg:approx-modexp`).

`reduce_exponent` and `_common` are shared by both versions; `delayed_mbu` is the version of
Appendix B, `immediate_mbu` the first version, which uncomputes each Lookup right after use.
"""
