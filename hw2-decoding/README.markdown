A phrase-based stack decoder implemented in Python, with arbitrary
reordering, future cost estimation, and histogram pruning.
For performance reasons, it's recommended that you run the decoder with
PyPy instead of the standard CPython interpreter:

    $ pypy decode.py TM LM INPUT

By default, the decoder allows any number of source words to be skipped
between adjacent target phrases, but uses a stack size of 1 and only
considers the single most likely translation for each source phrase as
given in the translation model.
To change this behavior, use the `-k`, `-r`, and `-s` options.
For more details, pass the `--help` option.

The `models` module, and portions of the main decoder, are taken from
[Adam Lopez's reference implementation][ref] for this assignment.

[ref]: https://github.com/alopez/en600.468/tree/master/decoder
