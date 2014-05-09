"""Core MEANT functions."""


from __future__ import division
from itertools import product
import logging

import munkres
from textblob import TextBlob


m = munkres.Munkres()


def costify(similarity_matrix):
    """Transform a similarity matrix into a cost matrix."""
    return munkres.make_cost_matrix(similarity_matrix, lambda s: 1 - s)


def similarity(one, two, lexsim):
    """Return the combined per-word pairwise similarity of the strings
    *one* and *two*."""
    one_words = TextBlob(one).words
    two_words = TextBlob(two).words
    word_similarities = [
        [lexsim.similarity(one_word.lower(), two_word.lower())
            for two_word in two_words]
        for one_word in one_words]
    word_alignments = m.compute(costify(word_similarities))
    return (sum(word_similarities[one_index][two_index]
                for one_index, two_index in word_alignments) /
            max(len(one_words), len(two_words)))


def score(hyp_frames, hyp_words, ref_frames, ref_words, lexsim):
    """Return the MEANT score for a hypothesis against a reference.
    The *frames* arguments are lists of semantic frame dictionaries as
    returned by ``pymeant.formats.assert_tagger.parse_line()``, while
    the *words* arguments are sequences of the words in each sentence.
    The *lexsim* argument is an object with a ``similarity()`` method
    that can be used to compute the lexical similarity of two words."""
    # Handle some edge cases.
    #
    # If both the hypothesis and reference have no frames, the F1 score
    # is 1.0.  Give the translators a pat on the back.
    if not hyp_frames and not ref_frames:
        logging.debug('No frames in either hypothesis or reference; '
                      'assigning score of 1.0')
        return 1.0
    # If exactly one of the hypothesis and reference has no frames,
    # either the precision or the recall is 0.0, so the F1 is also 0.0.
    if not hyp_frames or not ref_frames:
        logging.debug('Frames in one of hypothesis or reference, but '
                      'not the other; assigning score of 0.0')
        return 0.0

    # Align the frames to each other.
    frame_similarities = [
        [similarity(hyp_frame['TARGET'], ref_frame['TARGET'], lexsim)
            for ref_frame in ref_frames]
        for hyp_frame in hyp_frames]
    frame_alignments = m.compute(costify(frame_similarities))

    # "Possible" counts include the predicate and all arguments.
    #
    # TODO:  Implement weighted normalization.
    hyp_score = ref_score = 0.0
    hyp_possible = ref_possible = 0.0
    for hyp_index, ref_index in frame_alignments:
        hyp_frame = hyp_frames[hyp_index]
        ref_frame = ref_frames[ref_index]
        pred_similarity = frame_similarities[hyp_index][ref_index]
        logging.debug('Aligned hypothesis frame with predicate "%s" to '
                      'reference frame with predicate "%s" (%f)',
                      hyp_frame['TARGET'], ref_frame['TARGET'],
                      pred_similarity)
        hyp_coverage = (sum(len(x.split()) for x in hyp_frame.itervalues()) /
                        len(hyp_words))
        ref_coverage = (sum(len(x.split()) for x in ref_frame.itervalues()) /
                        len(ref_words))
        common_args = (frozenset(hyp_frame) & frozenset(ref_frame) -
                       frozenset(['TARGET']))
        arg_similarities = 0.0
        for arg in common_args:
            arg_similarity = similarity(hyp_frame[arg], ref_frame[arg],
                                        lexsim)
            logging.debug('Aligned hypothesis argument %s "%s" to '
                          'reference argument %s "%s" (%f)',
                          arg, hyp_frame[arg],
                          arg, ref_frame[arg], arg_similarity)
            arg_similarities += arg_similarity
        hyp_score += (hyp_coverage * (pred_similarity + arg_similarities) /
                      len(hyp_frame))
        ref_score += (ref_coverage * (pred_similarity + arg_similarities) /
                      len(ref_frame))
        hyp_possible += hyp_coverage
        ref_possible += ref_coverage

    precision = hyp_score / hyp_possible
    recall    = ref_score / ref_possible
    if not precision or not recall:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
