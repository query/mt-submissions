#!/usr/bin/env python
"""A word aligner for translation pairs."""

from __future__ import division
from collections import defaultdict, namedtuple
from itertools import chain, islice, izip, product
from math import log
import sys


class SentencePair(namedtuple('SentencePair', ['english', 'french'])):
    """A pair of sentences known to be translations of each other."""

    @classmethod
    def from_strings(cls, *args):
        # Use the hash of the string instead of the string itself, in
        # order to save ourselves a little memory.
        return super(SentencePair, cls).__new__(
            cls, *(map(hash, s.strip().split()) for s in args))


def align(pairs, iterations=5):
    """Train an IBM Model 2 aligner using EM for the given number of
    *iterations* on the sentence pairs in *pairs*, and return a list
    containing the most probable alignment vector for each pair.
    """
    # Initialize *t* and *q* uniformly.  The marginal probabilities here
    # won't add up to 1 at all, but that won't affect the results of EM.
    # *t* maps (f, e) to t(f | e).  *q* maps (j, i, l, m) to q(j | i, l,
    # m).
    t = defaultdict(lambda: 1.0)
    q = defaultdict(lambda: 1.0)

    # Perform expectation maximization.
    for iteration in range(iterations):
        # Various count tables.
        word_pairs = defaultdict(float)
        word_pair_marginals = defaultdict(float)
        positions = defaultdict(float)
        position_marginals = defaultdict(float)
        for k, pair in enumerate(pairs):
            if k % 500 == 0:
                progress = (iteration + k / len(pairs)) / iterations
                sys.stderr.write('\rPerforming EM... {:.1f}% complete'
                                 .format(progress * 100))
            l = len(pair.english)
            m = len(pair.french)
            for i, f in enumerate(pair.french):
                # Calculate the normalization term.
                norm = sum(t[(f, e)] * q[(j, i, l, m)]
                           for j, e in enumerate(pair.english))
                # Find expected counts.
                for j, e in enumerate(pair.english):
                    expected = t[(f, e)] * q[(j, i, l, m)] / norm
                    word_pairs[(f, e)] += expected
                    word_pair_marginals[e] += expected
                    positions[(j, i, l, m)] += expected
                    position_marginals[(i, l, m)] += expected
        t.clear()
        for f, e in word_pairs:
            t[(f, e)] = word_pairs[(f, e)] / word_pair_marginals[e]
        q.clear()
        for j, i, l, m in positions:
            q[(j, i, l, m)] = (positions[(j, i, l, m)] /
                               position_marginals[(i, l, m)])
    sys.stderr.write('\rPerforming EM... 100.0% complete, done.\n')

    # Return our decoding of the input sentence pairs.
    vectors = []
    for k, pair in enumerate(pairs):
        if k % 500 == 0:
            progress = k / len(pairs)
            sys.stderr.write('\rDecoding... {:.1f}% complete'
                             .format(progress * 100))
        l = len(pair.english)
        m = len(pair.french)
        vector = []
        for i, f in enumerate(pair.french):
            p_alignment = lambda j: (t[(f, pair.english[j])] *
                                     q[(j, i, l, m)])
            best_j = max(xrange(len(pair.english)), key=p_alignment)
            vector.append(best_j)
        vectors.append(vector)
    sys.stderr.write('\rDecoding... 100.0% complete, done.\n')
    return vectors


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='A word aligner for translation pairs.')
    parser.add_argument(
        'english_file', metavar='ENGLISH', type=argparse.FileType('r'),
        help='file containing English sentences')
    parser.add_argument(
        'french_file', metavar='FRENCH', type=argparse.FileType('r'),
        help='file containing French sentences')
    parser.add_argument(
        '-n', '--max-sentences', type=int,
        help='maximum number of sentences to train on')
    args = parser.parse_args()

    # Generate sentence pairs from the source files.
    pairs = [SentencePair.from_strings(*pair) for pair
             in islice(izip(args.english_file, args.french_file),
                       args.max_sentences)]
    print >> sys.stderr, 'Running forward alignment...'
    forward = align(pairs)

    for i in xrange(len(pairs)):
        pairs[i] = SentencePair(*reversed(pairs[i]))
    print >> sys.stderr, 'Running reverse alignment...'
    reverse = align(pairs)

    # Intersect the forward and reverse alignments.
    for forward_vector, reverse_vector in izip(forward, reverse):
        print ' '.join('{}-{}'.format(i, j)
                       for i, j in enumerate(forward_vector)
                       if reverse_vector[j] == i)


if __name__ == '__main__':
    main()
