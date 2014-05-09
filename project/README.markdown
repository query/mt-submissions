A Python implementation of an unparameterized version of [the MEANT
machine translation evaluation metric][meant].
Note that this version is a proof of concept not intended for use in
production systems.
In particular, scoring is very slow even on modest data sets.

To install, use `setup.py`:

    $ python setup.py install

Before scoring translation hypotheses, you will need to train a lexical
similarity model using `python -m pymeant train`.
A parser for Gigaword corpus files is included for convenience:

    $ python -m pymeant.formats.gigaword nyt199504.gz | python -m pymeant train - lexsim.pkl

To perform the actual scoring, use `python -m pymeant score`, passing in
the hypotheses and reference sentences as both plain text (one per line)
and [ASSERT][assert]-tagged parse files:

    $ python -m pymeant score lexsim.pkl hypotheses.{txt,parse} reference.{txt,parse}

For further information, pass the `--help` option.

[meant]: http://www.aclweb.org/anthology/W/W12/W12-3129.pdf
[assert]: http://cemantix.org/software/assert.html
