#!/usr/bin/env python
"""Inflect a lemmatized corpus."""


from __future__ import division, unicode_literals
from collections import Counter, defaultdict
import io
from itertools import izip
import os.path

from nltk.model.ngram import NgramModel
from nltk.probability import (ConditionalFreqDist, ConditionalProbDist,
                              LidstoneProbDist)

from tree import DepTree


def dep_ngrams(n, l_sentence, f_sentence, tree):
    """Yield *n*-grams decomposed over the dependency tree *tree*,
    with lemmata from *l_sentence* as context and inflected forms
    from *f_sentence* as the final word of each *n*-gram."""
    for lemma, form, node in izip(l_sentence, f_sentence, tree):
        # Try and get up to *n* nodes by traversing the tree.
        ngram = []
        current = node
        for i in xrange(n):
            if i == 0:
                ngram.insert(0, form)
            else:
                # Nodes are 1-indexed, so we have to subtract 1
                # to ensure that we've got the right word.
                ngram.insert(0, l_sentence[current.index() - 1])
            parent_index = node.parent_index()
            if parent_index == 0:
                break
            else:
                current = tree.node(parent_index)
        # Pad the *n*-gram to the right length.
        ngram = ('',) * (n - len(ngram)) + tuple(ngram)
        yield ngram


def lidstone_estimator(fdist, bins):
    return LidstoneProbDist(fdist, 0.2)


def utf8open(filename):
    return io.open(filename, encoding='utf-8')


class DependencyNgramModel(NgramModel):
    """An *n*-gram model that decomposes over dependency trees instead
    of in a left-to-right fashion."""

    def __init__(self, n, l_sentences, f_sentences, trees):
        self._n = n
        self._lpad = ('',) * (n - 1)
        self._rpad = ()

        estimator = lidstone_estimator

        cfd = ConditionalFreqDist()
        self._ngrams = set()

        for l_sentence, f_sentence, tree \
                in izip(l_sentences, f_sentences, trees):
            for ngram in dep_ngrams(n, l_sentence, f_sentence, tree):
                self._ngrams.add(ngram)
                context = ngram[:-1]
                token = ngram[-1]
                cfd[context].inc(token)

        self._model = ConditionalProbDist(cfd, estimator, len(cfd))

        if n > 1:
            self._backoff = DependencyNgramModel(
                n - 1, l_sentences, f_sentences, trees)


class Inflector(object):
    """A simple inflector based on a lemma bigram model."""

    def __init__(self, training_prefix):
        l_sentences = []
        f_sentences = []
        c_sentences = []
        trees = []
        # The set of possible inflections for each lemma.
        self.inflections = defaultdict(set)
        with utf8open(training_prefix + '.lemma') as lemma_file, \
             utf8open(training_prefix + '.form') as form_file, \
             utf8open(training_prefix + '.tree') as tree_file:
            for lemma_line, form_line, tree_line \
                    in izip(lemma_file, form_file, tree_file):
                l_sentence = lemma_line.split()
                f_sentence = form_line.split()
                c_sentence = []
                for lemma, form in izip(l_sentence, f_sentence):
                    c_sentence.append('{}~{}'.format(lemma, form))
                    self.inflections[lemma].add(form)
                l_sentences.append(l_sentence)
                f_sentences.append(f_sentence)
                c_sentences.append(c_sentence)
                trees.append(DepTree(tree_line))
        self.lr_model = NgramModel(2, c_sentences, pad_left=True,
                                   estimator=lidstone_estimator)
        self.dp_model = DependencyNgramModel(2, l_sentences,
                                             f_sentences, trees)

    def inflect(self, testing_prefix, dp_weight=0.5):
        """Return a list containing inflected versions of the sentences
        described by the files under *testing_prefix*."""
        lr_weight = 1 - dp_weight
        inflected = []
        with utf8open(testing_prefix + '.lemma') as lemma_file, \
             utf8open(testing_prefix + '.tree') as tree_file:
            for lemma_line, tree_line in izip(lemma_file, tree_file):
                l_sentence = lemma_line.split()
                tree = DepTree(tree_line)
                ngrams = dep_ngrams(2, l_sentence,
                                    l_sentence,  # not used here
                                    tree)
                forms = []
                last_lemma = None
                for lemma, dep_ngram in izip(l_sentence, ngrams):
                    if not self.inflections[lemma]:
                        # We've never seen this lemma before, so just
                        # output it as-is and move on.
                        forms.append(lemma)
                        continue
                    best_form = None
                    best_score = float('-inf')
                    for form in self.inflections[lemma]:
                        if last_lemma is None:
                            context = ['']
                        else:
                            context = ['{}~{}'.format(last_lemma, forms[-1])]
                        score = (
                            lr_weight * self.lr_model.prob(
                                '{}~{}'.format(lemma, form), context) +
                            dp_weight * self.dp_model.prob(
                                form, dep_ngram[:-1]))
                        if score > best_score:
                            best_form = form
                            best_score = score
                    forms.append(best_form)
                    last_lemma = lemma
                inflected.append(' '.join(forms))
        return inflected


def main():
    """Command line entry point."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__)
    parser.add_argument(
        'training_prefix', metavar='TRAIN',
        help='training data path prefix (e.g. "data/train")')
    parser.add_argument(
        'testing_prefixes', metavar='TEST', nargs='*',
        help='testing data path prefix (e.g. "data/dtest")')
    parser.add_argument(
        '-d', '--dep-weight', type=float, default=0.5,
        help='weight of dependency n-gram feature (default 0.5)')
    args = parser.parse_args()

    inflector = Inflector(args.training_prefix)

    for testing_prefix in args.testing_prefixes:
        for sentence in inflector.inflect(testing_prefix,
                                          dp_weight=args.dep_weight):
            print sentence.encode('utf-8')


if __name__ == '__main__':
    main()
