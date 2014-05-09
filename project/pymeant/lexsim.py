"""Lexical similarity metrics."""


from __future__ import division
from collections import Counter, defaultdict


def _key(one, two):
    return ' '.join(sorted([one, two]))


class Jaccard(object):
    """An implementation of the Jaccard similarity metric, as described
    in (Tumuluru et al., 2012), section 3.5."""

    def __init__(self, window_size=3):
        self.window_size = window_size
        """The window size used to look for lexical cooccurrences."""

        self.joint = Counter()
        """This metric's joint cooccurrence table."""

        self.contexts = defaultdict(set)
        """A dictionary mapping each known word to the set of other
        words that it has cooccurred with."""

    def update(self, words):
        """Add the sentence composed of the sequence of *words* to this
        metric's count table."""
        for start, one in enumerate(words):
            end = start + self.window_size
            # Note that the slicing syntax means that our window does
            # *not* include *end*, which is what we want.  Otherwise,
            # the window would be one longer than our window size.
            pairs = set()
            for two in words[start + 1:end]:
                pairs.add(_key(one, two))
                self.contexts[one].add(two)
                self.contexts[two].add(one)
            # Only add 1 to the cooccurrence count per pair per window,
            # even if the pair occurs more than once.
            for pair in pairs:
                self.joint[pair] += 1

    def similarity(self, one, two):
        """Return a float value indicating the similarity of the
        contexts in which *one* and *two* have been seen, ranging from
        0.0 (the two words have no common context) to 1.0 (the two words
        always appear in the same context)."""
        if one == two:
            return 1.0
        minima = maxima = 0.0
        for context in (self.contexts[one] | self.contexts[two]):
            one_context = _key(one, context)
            two_context = _key(two, context)
            minima += min(self.joint[one_context], self.joint[two_context])
            maxima += max(self.joint[one_context], self.joint[two_context])
        if not maxima:
            return 0.0
        return minima / maxima
