#!/usr/bin/env python
"""A translation decoder."""

from collections import defaultdict, namedtuple

import models


Hypothesis = namedtuple('Hypothesis',
                        ['logprob', 'future_cost', 'coverage',
                         'lm_state', 'predecessor', 'candidate'])


def decode(tm, lm, source_sentence,
           stack_size=1, max_reordering=None):
    """Return the most probable decoding of *source_sentence* under the
    provided probabilistic translation and language models."""
    # Compute the future cost table.
    future_costs = {}
    for segment_length in xrange(1, len(source_sentence) + 1):
        for start in xrange(len(source_sentence) - segment_length + 1):
            end = start + segment_length
            future_costs[(start, end)] = float('-inf')
            candidates = tm.get(source_sentence[start:end], [])
            if candidates:
                logprob = candidates[0].logprob
                lm_state = tuple()
                for target_word in candidates[0].english.split():
                    lm_state, word_logprob = lm.score(lm_state, target_word)
                    logprob += word_logprob
                future_costs[(start, end)] = logprob
            for mid in xrange(start + 1, end):
                future_costs[(start, end)] = max(
                    future_costs[(start, mid)] + future_costs[(mid, end)],
                    future_costs[(start, end)])
    # Actually start decoding.
    initial = Hypothesis(0.0, future_costs[(0, len(source_sentence))],
                         (False,) * len(source_sentence),
                         lm.begin(), None, None)
    # We add 1 here because we need to have stacks for both ends: 0 and
    # len(source_sentence).
    stacks = [{} for _ in xrange(len(source_sentence) + 1)]
    stacks[0][lm.begin()] = initial
    # Iterate over every stack but the last.  It's not possible to add
    # anything to a hypothesis in the last stack anyway, so we skip it.
    for i, stack in enumerate(stacks[:-1]):
        # Take only the best *stack_size* hypotheses.  Using the sum of
        # the log-probability and the future cost negatively impacts the
        # model score (??).
        hypotheses = sorted(stack.itervalues(),
                            key=lambda h: -h.logprob)[:stack_size]
        for hypothesis in hypotheses:
            # Save ourselves a couple of levels of indentation later on.
            def untranslated_segments():
                if max_reordering is None:
                    starts = xrange(len(source_sentence))
                else:
                    starts = xrange(min(i + max_reordering,
                                        len(source_sentence)))
                for start in starts:
                    if hypothesis.coverage[start]:
                        continue
                    ends = xrange(start, len(source_sentence))
                    for end in ends:
                        if hypothesis.coverage[end]:
                            break
                        yield (start, end + 1)
            # Iterate over blocks of untranslated source words.
            for start, end in untranslated_segments():
                source_phrase = source_sentence[start:end]
                # Get all of the potential candidate translations.
                candidates = tm.get(source_phrase, [])
                # Translate unknown unigrams to themselves.
                if not candidates and len(source_phrase) == 1:
                    candidates.append(models.phrase(source_phrase[0], 0.0))
                for candidate in candidates:
                    logprob = hypothesis.logprob + candidate.logprob
                    # Make a new coverage vector with the appropriate
                    # elements set to True.  This isn't pretty.  Sorry.
                    coverage = (hypothesis.coverage[:start] +
                                (True,) * (end - start) +
                                hypothesis.coverage[end:])
                    # Find the future cost estimate for this hypothesis
                    # by summing over contiguous incomplete segments.
                    future_cost = 0.0
                    cost_start = None
                    for cost_i, covered in enumerate(coverage + (True,)):
                        if covered:
                            if cost_start is not None:
                                future_cost += \
                                    future_costs[(cost_start, cost_i)]
                            cost_start = None
                        else:
                            if cost_start is None:
                                cost_start = cost_i
                    # Make a new LM state.
                    lm_state = hypothesis.lm_state
                    for target_word in candidate.english.split():
                        lm_state, word_logprob = \
                            lm.score(lm_state, target_word)
                        logprob += word_logprob
                    # Add the final transition probability if the end of
                    # this segment is also the end of the sentence.
                    if end == len(source_sentence):
                        logprob += lm.end(lm_state)
                    # If the new hypothesis is the best hypothesis for
                    # its state and number of completed words, push it
                    # onto the stack, replacing any that is present.
                    completed = sum(int(x) for x in coverage)
                    if (lm_state not in stacks[completed] or
                            (stacks[completed][lm_state].logprob +
                             stacks[completed][lm_state].future_cost) <
                            logprob + future_cost):
                        stacks[completed][lm_state] = Hypothesis(
                            logprob, future_cost, coverage,
                            lm_state, hypothesis, candidate)
    # We don't need to specify a key, since we're looking for the best
    # log-probability, and that's the first element of a hypothesis.
    best = max(stacks[-1].itervalues())
    current = best
    decoding = []
    while current.candidate:
        decoding.insert(0, current.candidate.english)
        current = current.predecessor
    return tuple(decoding)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='A translation decoder.')
    parser.add_argument(
        'tm_path', metavar='TM',
        help='path to translation model')
    parser.add_argument(
        'lm_path', metavar='LM',
        help='path to language model')
    parser.add_argument(
        'input_file', metavar='INPUT', type=argparse.FileType('r'),
        help='path to file containing sentences to decode')
    parser.add_argument(
        '-k', '--max-candidates', type=int, default=1,
        help='maximum number of translation candidates to consider for '
             'each phrase')
    parser.add_argument(
        '-r', '--max-reordering', type=int,
        help='maximum number of source words that can be skipped '
             'during reordering')
    parser.add_argument(
        '-s', '--stack-size', type=int, default=1,
        help='maximum hypothesis stack size')
    args = parser.parse_args()

    tm = models.TM(args.tm_path, args.max_candidates)
    lm = models.LM(args.lm_path)
    for source_line in args.input_file:
        source_sentence = tuple(source_line.split())
        print ' '.join(decode(tm, lm, source_sentence,
                              stack_size=args.stack_size,
                              max_reordering=args.max_reordering))


if __name__ == '__main__':
    main()
