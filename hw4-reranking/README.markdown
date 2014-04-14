A reranker for *k*-best translation hypothesis lists using the PRO
algorithm (Hopkins & May, 2011).
Pairwise rankings are generated on the sentence-wise BLEU score, with
training features based on TM and LM probabilities, word count, verb
count, and the number of apparently untranslated tokens.
These rankings are then fed into a linear SVM, and the weights used to
rescore test data.

Before running the reranker, you must install its dependencies:

    $ pip install -r requirements.txt
    $ python rerank.py DATA_DIR

For information on optional settings, pass the `--help` option.

The `bleu` module is taken from [Adam Lopez's reference
implementation][ref] for this assignment.

[ref]: https://github.com/alopez/en600.468/tree/master/reranker
