"""Tools for working with ASSERT output files."""

# This module isn't called "assert" because that's a Python keyword.


import re


ARGUMENT_TAG = re.compile(r'\[(?P<label>.+?) (?P<contents>.+?)\s*\]')


def parse_line(line):
    """Parse a single line of ASSERT output, and return a dictionary
    mapping semantic argument labels to their contents."""
    return {m.group('label'): m.group('contents')
            for m in ARGUMENT_TAG.finditer(line)}


def parse_file(assert_file):
    """Parse an ASSERT output file, and return a dictionary mapping each
    sentence's numeric ID to a list of dictionaries representing its
    frames, as returned by ``parse_line()``."""
    sentences = {}
    for line in assert_file:
        number, _, line = line.partition(': ')
        sentences.setdefault(int(number), []).append(parse_line(line))
    return sentences
