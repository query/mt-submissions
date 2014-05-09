"""Main CLI frontend."""


from itertools import izip
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys

from textblob import TextBlob

from .formats import assert_tagger, gigaword
from .lexsim import Jaccard
from .meant import score as meant_score


def train(args):
    """Entry point for the "train" subcommand."""
    lexsim = Jaccard(window_size=args.window_size)
    for line in args.training_file:
        blob = TextBlob(line)
        for sentence in blob.sentences:
            lexsim.update([word.lower() for word in sentence.words])
    pickle.dump(lexsim, args.lexsim_file)


def score(args):
    """Entry point for the "score" subcommand."""
    lexsim = pickle.load(args.lexsim_file)
    hyp_sentences = assert_tagger.parse_file(args.hyp_parses_file)
    ref_sentences = assert_tagger.parse_file(args.ref_parses_file)
    for number, (hyp_line, ref_line) in enumerate(izip(
            args.hyp_text_file, args.ref_text_file)):
        hyp_sentence = hyp_sentences.get(number, [])
        ref_sentence = ref_sentences.get(number, [])
        print meant_score(hyp_sentence, hyp_line.split(),
                          ref_sentence, ref_line.split(),
                          lexsim)


def main():
    """Command line entry point."""
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__)
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='enable debugging output')
    subparsers = parser.add_subparsers(metavar='SUBCOMMAND')

    train_parser = subparsers.add_parser(
        'train', help='train lexical similarity model')
    train_parser.set_defaults(func=train)
    train_parser.add_argument(
        'training_file', metavar='TRAIN',
        type=argparse.FileType('r'),
        help='path to training file')
    train_parser.add_argument(
        'lexsim_file', metavar='LEXSIM',
        type=argparse.FileType('wb'),
        help='destination path for lexical similarity count table')
    train_parser.add_argument(
        '-w', '--window-size', metavar='W',
        type=int, default=3,
        help='co-occurrence window size')

    score_parser = subparsers.add_parser(
        'score', help='score hypotheses against reference data')
    score_parser.set_defaults(func=score)
    score_parser.add_argument(
        'lexsim_file', metavar='LEXSIM',
        type=argparse.FileType('rb'),
        help='path to lexical similarity count table')
    score_parser.add_argument(
        'hyp_text_file', metavar='HYP_TEXT',
        type=argparse.FileType('r'),
        help='path to hypothesis text file')
    score_parser.add_argument(
        'hyp_parses_file', metavar='HYP_PARSES',
        type=argparse.FileType('r'),
        help='path to hypothesis parse file')
    score_parser.add_argument(
        'ref_text_file', metavar='REF_TEXT',
        type=argparse.FileType('r'),
        help='path to reference text file')
    score_parser.add_argument(
        'ref_parses_file', metavar='REF_PARSES',
        type=argparse.FileType('r'),
        help='path to reference parse file')

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    args.func(args)


if __name__ == '__main__':
    main()
