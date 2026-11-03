"""Cross-realization GRPO (super-group training).

One training unit is a *semantic task* s realized K ways; each realization gets M
rollouts. Outcome advantage is normalized WITHIN a realization (so realization
difficulty cannot contaminate credit assignment); a grounding/invariance
advantage is normalized ACROSS realizations of the same semantic task.

    A = A_outcome(within realization)  +  lambda * A_grounding(across realizations)

Design + ablations: docs/eidf/cross-realization-grpo.md
"""
