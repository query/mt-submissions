#!/usr/bin/env python
"""A reranker for k-best translation hypothesis lists."""


from __future__ import division
from collections import Counter
import heapq
from itertools import islice
import logging
import operator
import os.path
import random

import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from textblob import TextBlob

import bleu


K_BEST = 100
"""The number of hypotheses per sentence in the k-best data file."""


def featurize(hyp_line, src_line):
    """Return a feature vector for a translation hypothesis based on the
    data in *hyp_line* and *src_line*."""
    fv = []
    _, hyp, probs = hyp_line.split(' ||| ')
    _, src = src_line.split(' ||| ')
    hyp_blob = TextBlob(hyp.decode('utf-8'))
    src_blob = TextBlob(src.decode('utf-8'))
    # Feature class 1:  TM and LM probabilities.
    for prob in probs.split():
        _, _, v = prob.partition('=')
        fv.append(float(v))
    # Feature class 2:  Various word counts.
    fv.append(len(hyp_blob.words))
    fv.append(sum(pos.startswith('V') for _, pos in hyp_blob.tags))
    # Feature class 3:  Number of apparently untranslated words.
    hyp_words = Counter(hyp_blob.words)
    src_words = Counter(src_blob.words)
    fv.append(sum((hyp_words & src_words).itervalues()))
    return fv


def gold_score(hyp_line, ref_line):
    """Return the gold score for a translation hypothesis based on the
    data in *hyp_line* and *ref_line*."""
    _, hyp, _ = hyp_line.split(' ||| ')
    hyp_words = hyp.split()
    ref_words = ref_line.split()
    return bleu.bleu(tuple(bleu.bleu_stats(hyp_words, ref_words)))


def rank(data_dir, gamma=100, xi=100):
    """Train a pairwise ranking classifier on the training data in
    *data_dir*, and return its judgment of the single best hypothesis
    for each sentence on the development and testing data.  The *gamma*
    and *xi* parameters control training data sampling, as in (Hopkins
    & May, 2011)."""
    logging.info('Sampling from training data')
    train_data = []  # Each element contains features, then a label.
    with open(os.path.join(data_dir, 'train.ref')) as ref_file, \
         open(os.path.join(data_dir, 'train.src')) as src_file, \
         open(os.path.join(data_dir, 'train.100best')) as hyp_file:
        for i, src_line in enumerate(src_file):
            if i > 0 and i % 50 == 0:
                logging.info('... finished %d sentences', i)
            ref_line = next(ref_file)
            hyp_lines = list(islice(hyp_file, K_BEST))
            # Sample as in (Hopkins & May, 2011), figure 4.
            sentence_train_data = []
            for _ in xrange(gamma):
                one_line, two_line = random.sample(hyp_lines, 2)
                delta_g = (gold_score(one_line, ref_line) -
                           gold_score(two_line, ref_line))
                # Here we use the step function in section 4.2 to
                # determine whether we'll consider this pair.
                if abs(delta_g) < 0.05:
                    continue
                # Build a single unbalanced sample for this hypothesis
                # pair.  The corresponding sample of the opposite label
                # will be constructed later.  The absolute difference is
                # up front because that's what our heap is sorted by.
                one_fv = featurize(one_line, src_line)
                two_fv = featurize(two_line, src_line)
                sample = (abs(delta_g), one_fv, two_fv, delta_g > 0)
                # Push 'em onto the heap.
                assert len(sentence_train_data) <= xi
                if len(sentence_train_data) == xi:
                    heapq.heappushpop(sentence_train_data, sample)
                else:
                    heapq.heappush(sentence_train_data, sample)
            train_data.extend(sentence_train_data)
    # Build our balanced samples.
    logging.info('Balancing %d samples', len(train_data))
    train_fvs = []
    train_labels = []
    for delta_g, one_fv, two_fv, label in train_data:
        train_fvs.append(map(operator.sub, one_fv, two_fv))
        train_labels.append(label)
        train_fvs.append(map(operator.sub, two_fv, one_fv))
        train_labels.append(not label)
    # Now that we have our data, build our classifier.
    logging.info('Training binary classifier')
    scaler = StandardScaler()
    scaler.fit(train_fvs)
    train_fvs = scaler.transform(train_fvs)
    clf = GridSearchCV(LinearSVC(),
                       {'C': 10.0 ** np.arange(-2, 4)},
                       cv=5)  # Merely importing TextBlob causes a hang
                              # in multiprocessing (n_jobs=-1).  ???
    clf.fit(train_fvs, train_labels)
    logging.info('Best cross-validation hyperparameters: %s',
                 ', '.join('{}={}'.format(k, v)
                           for k, v in clf.best_params_.iteritems()))
    logging.info('Reranking testing data')
    best_hyps = []
    with open(os.path.join(data_dir, 'dev+test.src')) as src_file, \
         open(os.path.join(data_dir, 'dev+test.100best')) as hyp_file:
        for src_line in src_file:
            hyp_lines = list(islice(hyp_file, K_BEST))
            hyp_fvs = scaler.transform([featurize(hyp_line, src_line)
                                        for hyp_line in hyp_lines])
            # Grab the coefficients out of the classifier and use them
            # to compute the scores for each hypothesis, then pluck the
            # best one from the list.
            #
            # <https://gist.github.com/agramfort/2071994>
            best_i = np.argmax(np.dot(hyp_fvs, clf.best_estimator_.coef_.T))
            best_hyps.append(hyp_lines[best_i].split(' ||| ')[1])
    return best_hyps


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_dir', metavar='DATA_DIR',
        help='path to directory containing train/dev/test data')
    parser.add_argument(
        '-g', '--gamma', type=int, default=100,
        help='number of pairwise samples to consider for each sentence')
    parser.add_argument(
        '-x', '--xi', type=int, default=100,
        help='maximum number of samples to keep for each sentence')
    args = parser.parse_args()

    logging.basicConfig(format='%(message)s', level=logging.INFO)

    for best_hyp in rank(args.data_dir, gamma=args.gamma, xi=args.xi):
        print best_hyp


if __name__ == '__main__':
    main()
