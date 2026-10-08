"""Concrete Scorer implementations.

geval_scorers.py -- hand-written G-Eval-style scorers (no external eval
library required; these are what the default registry uses out of the box).

deepeval_scorers.py -- thin wrappers around DeepEval's validated RAG
metrics, for the criteria it already covers well (faithfulness, contextual
recall/precision, relevancy). Requires `pip install deepeval`.
"""
