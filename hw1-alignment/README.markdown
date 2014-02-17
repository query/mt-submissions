A Python implementation of IBM Model 2, which trains forward and reverse
models and intersects the resulting alignments.
For performance reasons, it's recommended that you run the aligner with
PyPy instead of the standard CPython interpreter:

    $ pypy align.py ENGLISH FRENCH

For more details, pass the `--help` option.
