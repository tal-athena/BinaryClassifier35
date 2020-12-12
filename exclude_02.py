"""
Include/exclude dimensions from the index.

This script only updates the Dimensions table.  It doesn't remove anything from
the DocumentsToDimensions table.  If you want to remove excluded dimensions
from the DocumentsToDimensions table, use prune.py.
"""

import sqlite3
import sys
import spacy

import util
from calc_dim_01 import match_exclude_regex

from functools import reduce

def create_parser(usage):
    """Create an object to use for the parsing of command-line arguments."""
    from optparse import OptionParser
    parser = OptionParser(usage)
    parser.add_option(
            '--debug', 
            '-d', 
            dest='debug', 
            default=False,
            action='store_true',
            help='Show debug information')
    parser.add_option(
            '--include', 
            '-i', 
            dest='include', 
            default=False,
            action='store_true',
            help='Include instead of excluding')
    parser.add_option(
            '--range',
            '-r',
            dest='range',
            default=None,
            help='Include/exclude a single dimension')
    parser.add_option(
            '--one',
            '-o',
            dest='one',
            default=None,
            type='int',
            help='Include/exclude a range of dimensions separated by a hyphen')
    parser.add_option(
            '--stopwords',
            '-s',
            dest='stopwords',
            action='store_true',
            default=False,
            help='Include/exclude common English stopwords')
    parser.add_option(
            '--regex',
            '-x',
            dest='regex',
            action='store_true',
            default=False,
            help='Maintain consistency with ExclusionRegex table')
    parser.add_option(
            '--numeric',
            '-n',
            dest='numeric',
            action='store_true',
            default=False,
            help='Exclude/include dimensions that are numeric')
    parser.add_option(
            '--shorter-than',
            '-S',
            dest='shorter_than',
            type='int',
            default=0,
            help='Exclude/include dimensions that are shorter than this')
    parser.add_option(
            '--non-alpha',
            dest='non_alpha',
            action='store_true',
            default=False,
            help='Exclude/include dimensions that are completely non-alphabetic')
    parser.add_option(
            '--non-alpha-partial',
            dest='non_alpha_partial',
            action='store_true',
            default=False,
            help='Exclude/include dimensions that are partially non-alphabetic')
    return parser

def process_term(c, term, pos, exclude):
    """
    Find the dimension that corresponds to the term and part of speech.
    Set its Exclude field to be equal to the specified value.
    """
    c.execute(
        'UPDATE Dimensions SET Exclude = ? WHERE Term = ? AND PartOfSpeech = ?',
        (exclude, term, pos))

def process_dimension(c, dim, exclude):
    """
    Set the Exclude field of the specified dimension to the specified value.
    """
    cmd = 'UPDATE Dimensions SET Exclude = ? WHERE DimensionId = ?'
    c.execute(cmd, (exclude, dim))

def exclude_regex(path):
    """Exclude all dimensions that match an exclusion regex."""
    conn = sqlite3.connect(path)
    c = conn.cursor()
    exclusion_regex = util.get_all_exclude_regex(c)
    all_dim = util.get_dimensions(c)
    cmd = 'UPDATE Dimensions SET Exclude = ? WHERE DimensionId = ?'
    for did, term, pos, exclude in all_dim:
        should_exclude = int(match_exclude_regex(
            pos, term, exclusion_regex))
        if exclude != should_exclude:
            c.execute(cmd, (should_exclude, did))
    c.close()
    conn.commit()
    conn.close()


def exclude_stopwords_spacy(nlp, path, language = "fr", include=False):
    """
    Exclude all dimensions that correspond to an internal list of    
    """
    
    conn = sqlite3.connect(path)
    c = conn.cursor()

    if language == 'fr':
        spacy_stopwords = spacy.lang.fr.stop_words.STOP_WORDS
    elif language == 'de':
        spacy_stopwords = spacy.lang.de.stop_words.STOP_WORDS
    elif language == 'en':
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    else :
        spacy_stopwords = []

    for stop_word in spacy_stopwords:
        tokens = nlp(stop_word)
        process_term(c, tokens[0].lemma_, tokens[0].pos_, not include)

    c.close()
    conn.commit()
    conn.close()

def exclude_range(path, start, end, include=False):
    """Exclude a range of dimensions."""
    conn = sqlite3.connect(path)
    c = conn.cursor()
    for i in range(start, end+1):
        process_dimension(c, i, not include)
    c.close()
    conn.commit()
    conn.close()

def exclude_numeric(path, include=False):
    """Exclude all dimensions that are numeric."""
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""UPDATE Dimensions SET Exclude = ?
            WHERE PartOfSpeech = 'CD'""",
            (0 if include else 1,))
    c.close()
    conn.commit()
    conn.close()

def exclude_unigrams_shorter_than(path, N, include=False):
    """Exclude all dimensions with term shorter than N characters."""
    conn = sqlite3.connect(path)
    c = conn.cursor()
    short_dim = [ d for d in util.get_dimensions(c) if len(d[1]) < N and len(d[1].split()) <= 1]
    for did, term, _, _ in short_dim:
        c.execute("""UPDATE Dimensions SET Exclude = ?
                WHERE DimensionId = ?""",
                (0 if include else 1, did))
    c.close()
    conn.commit()
    conn.close()

def exclude_ngrams_shorter_than(path, N, include=False):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    match = [ d[0] for d in util.get_dimensions(c)
            if not is_ngram_longer_than(d[1], N) ]
    for m in match:
        process_dimension(c, m, not include)
    c.close()
    conn.commit()
    conn.close()

def exclude_shorter_than(path, N, include=False):
    """Exclude all dimensions with term shorter than N characters."""
    conn = sqlite3.connect(path)
    c = conn.cursor()
    short_dim = [ d for d in util.get_dimensions(c) if len(d[1]) < N ]
    for did, term, _, _ in short_dim:
        c.execute("""UPDATE Dimensions SET Exclude = ?
                WHERE DimensionId = ?""",
                (0 if include else 1, did))
    c.close()
    conn.commit()
    conn.close()

def exclude_non_alpha_partial(path, include=False):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    match = [ d[0] for d in util.get_dimensions(c)
            if not is_complete_alpha_str_ngrams(d[1]) ]
    for m in match:
        process_dimension(c, m, not include)
    c.close()
    conn.commit()
    conn.close()

def exclude_non_alpha(path, include=False):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    match = [ d[0] for d in util.get_dimensions(c) if not is_alpha_str(d[1]) ]
    for m in match:
        process_dimension(c, m, not include)
    c.close()
    conn.commit()
    conn.close()

def is_alpha_str(s):
    return reduce(lambda x, y: x or y, [c.isalpha() for c in s])

def is_ngram_longer_than(s, N):
    strings = s.split()
    if len(strings) <= 1:
        return True # not ngram
    return reduce(lambda x, y: x and y, [len(c) > N for c in strings])

def is_complete_alpha_str_ngrams(s):
    strings = s.split()
    return reduce(lambda x, y: x and y, [is_complete_alpha_str(c) for c in strings])

def is_complete_alpha_str(s):
    return reduce(lambda x, y: x and y, [c.isalpha() for c in s])

def main():
    parser = create_parser('usage: %s file.sqlite3 [options]' % __file__)
    options, args = parser.parse_args()
    if len(args) != 1:
        parser.error('invalid number of arguments')

    count = 0
    if options.regex:
        count += 1
    elif options.stopwords:
        count += 1
    elif options.range:
        count += 1
    elif options.one is not None:
        count += 1
    elif options.numeric:
        count += 1
    elif options.shorter_than:
        count += 1
    elif options.non_alpha:
        count += 1
    elif options.non_alpha_partial:
        count += 1

    if count == 0:
        parser.error('no exclusion option selected')
    elif count > 1:
        parser.error('too many exclusion options selected')

    if options.regex:
        exclude_regex(args[0])
    elif options.stopwords:
        exclude_stopwords(args[0], options.include)
    elif options.range:
        start, end = list(map(int, options.range.split('-')))
        exclude_range(args[0], start, end, options.include)
    elif options.one is not None:
        process_dimension(args[0], options.one, options.include)
    elif options.numeric:
        exclude_numeric(args[0], options.include)
    elif options.shorter_than:
        exclude_shorter_than(args[0], options.shorter_than, options.include)
    elif options.non_alpha_partial:
        exclude_non_alpha_partial(args[0])
    elif options.non_alpha:
        exclude_non_alpha(args[0])
        
    return 0

if __name__ == '__main__':
    sys.exit(main())
