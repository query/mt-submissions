An inflector for lemmatized corpora using features from bigrams taken
linearly and over dependency structures.

Before running the inflector, you must install its dependencies:

    $ pip install -r requirements.txt
    $ python inflect.py TRAIN_PREFIX TEST_PREFIX

To change the relative weight of the dependency bigram features, use the
`-d` option with a value between 0 and 1 inclusive.
For further information, pass the `--help` option.

The `tree` module is taken from [Matt Post's reference
implementation][ref] for this assignment.

[ref]: https://github.com/mjpost/inflect
