An evaluator that performs pairwise ranking of translation hypotheses.
Features based on *n*-gram precision and recall, gzip-compressed byte
length, and word count are used to train an SVM classifier.
The maximum *n*-gram length is user-specifiable.
Optionally, combined word/POS features can also be added, or punctuation
stripped from input words.

Before running the evaluator, you must install its dependencies:

    $ pip install -r requirements.txt
    $ python evaluate.py HYPOTHESES LABELS

By default, the evaluator runs with a maximum *n*-gram length of 1.
For information on how to change these settings,
pass the `--help` option.
