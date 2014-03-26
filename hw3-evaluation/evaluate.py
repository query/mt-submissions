#!/usr/bin/env python
"""An evaluator for translation hypotheses."""


from __future__ import division
from collections import Counter, namedtuple
from gzip import GzipFile
import re
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from numpy import arange, array
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from textblob import TextBlob


#
# Utility functions and classes.
#

Sentence = namedtuple('Sentence', ['one', 'two', 'ref'])
"""Represents a pair of translation hypotheses and a reference
translation for some sentence."""


def cardinality(counter):
    """Return the cardinality of the multiset *counter*."""
    return sum(counter.itervalues())


def ngrams(iterable, n):
    """Yield successive *n*-grams from *iterable*."""
    if n > len(iterable):
        return
    for i in xrange(len(iterable) - n):
        yield tuple(iterable[i:i+n])


#
# Feature functions.
#

def gzip(sentence):
    """Return a tuple consisting of the gzip-compressed length of a
    file containing both the hypothesis and reference for each of the
    hypotheses in *sentence*."""
    def _gzip(hypothesis):
        compressed = StringIO()
        gzfile = GzipFile(mode='wb', fileobj=compressed)
        gzfile.write(u'\n'.join(hypothesis + sentence.ref).encode('utf-8'))
        gzfile.close()
        return len(compressed.getvalue())
    return (_gzip(sentence.one), _gzip(sentence.two))


def ngram_stats(sentence, ngram_length=1):
    """Return a tuple consisting of the *n*-gram precision, recall, and
    F1 scores for each of the hypotheses in *sentence*."""
    ref_set = Counter(ngrams(sentence.ref, ngram_length))
    def _ngram_stats(hypothesis):
        hyp_set = Counter(ngrams(hypothesis, ngram_length))
        if cardinality(hyp_set) == 0:
            precision = 0.0
        else:
            precision = cardinality(hyp_set & ref_set) / cardinality(hyp_set)
        if cardinality(ref_set) == 0:
            recall = 0.0
        else:
            recall = cardinality(hyp_set & ref_set) / cardinality(ref_set)
        if (precision + recall) == 0.0:
            return (0.0, 0.0, 0.0)
        f1_score = 2 * precision * recall / (precision + recall)
        return (precision, recall, f1_score)
    return (_ngram_stats(sentence.one) + _ngram_stats(sentence.two))


def pos_ngram_stats(sentence, *args, **kwargs):
    """Return a tuple consisting of the *n*-gram precision, recall, and
    F1 scores over a combination of the words and part-of-speech tags
    for each of the hypotheses in *sentence*."""
    def _tags(hypothesis):
        return TextBlob(u' '.join(hypothesis)).tags
    return ngram_stats(Sentence(*(_tags(h) for h in sentence)),
                       *args, **kwargs)


def word_counts(sentence):
    """Return a tuple consisting of the word counts for each of the
    hypotheses in *sentence*."""
    return (len(sentence.one), len(sentence.two))


#
# Classification mechanics.
#

def extract_features(hypothesis_file,
                     max_ngram_length=1,
                     pos_tagging=False,
                     strip_punctuation=False):
    """Return a NumPy array containing feature vectors computed for the
    hypotheses in *hypothesis_file*."""
    features = []
    for hypothesis_line in hypothesis_file:
        sentence_features = []
        if strip_punctuation:
            preprocess = lambda s: re.sub(r'\W+', '', s, flags=re.UNICODE)
        else:
            preprocess = lambda s: s
        sentence = Sentence(*([preprocess(w)
                               for w in s.decode('utf-8').split()]
                              for s in hypothesis_line.split('|||')))
        sentence_features.extend(gzip(sentence))
        sentence_features.extend(word_counts(sentence))
        for n in xrange(max_ngram_length):
            sentence_features.extend(ngram_stats(sentence, n))
            if pos_tagging:
                sentence_features.extend(pos_ngram_stats(sentence, n))
        features.append(sentence_features)
    return array(features)


def load_labels(label_file):
    """Return a NumPy array containing the labels in *labels_file*."""
    return array([int(label_line) for label_line in label_file])


def classify(features, labels, folds=5):
    """Perform supervised classification of the sentences described by
    *features* using the provided *labels*, and return a NumPy array
    with the classifier's predictions."""
    clf = GridSearchCV(SVC(), {'C': 10.0 ** arange(-2, 4)},
                       cv=folds, n_jobs=-1)
    clf.fit(features[:labels.shape[0]], labels)
    return clf.predict(features)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='An evaluator for translation hypotheses.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'hypothesis_file', metavar='HYPOTHESES', type=argparse.FileType('r'),
        help='path to file containing hypotheses')
    parser.add_argument(
        'label_file', metavar='LABELS', type=argparse.FileType('r'),
        help='path to file containing labels')
    parser.add_argument(
        '-f', '--folds', metavar='F', type=int, default=5,
        help='number of folds to use for cross-validation')
    parser.add_argument(
        '-n', '--max-ngram-length', metavar='N', type=int, default=1,
        help='consider n-grams up to length N for features')
    parser.add_argument(
        '-p', '--pos-tagging', action='store_true',
        help='add combined word/POS n-gram features')
    parser.add_argument(
        '-s', '--strip-punctuation', action='store_true',
        help='strip punctuation from words in each hypothesis')
    args = parser.parse_args()

    features = extract_features(args.hypothesis_file,
                                max_ngram_length=args.max_ngram_length,
                                pos_tagging=args.pos_tagging,
                                strip_punctuation=args.strip_punctuation)
    labels = load_labels(args.label_file)
    predictions = classify(features, labels)
    for prediction in predictions:
        print prediction


if __name__ == '__main__':
    main()
