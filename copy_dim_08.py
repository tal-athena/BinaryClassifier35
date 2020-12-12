"""Classify documents in an unlabeled database."""

import sqlite3
import sys
import nltk
import util

from math import log10
from calc_dim_01 import init_dim

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
    return parser

def copy_dim(src, dst):
    """
    Copy dimension information from conn1 into conn2.  Any existing information
    is destroyed.
    """
    conn1 = sqlite3.connect(src)
    conn2 = sqlite3.connect(dst)

    c1 = conn1.cursor()
    c2 = conn2.cursor()
    util.drop_tables(c2, [ 'Dimensions', 'DocumentsToDimensions' ])
    init_dim(c2)
    
    c1.execute("""SELECT DimensionId, Term, PartOfSpeech, Exclude, IDF, MRMR
            FROM Dimensions WHERE Exclude = 0""")
    for row in c1:
        c2.execute('INSERT INTO Dimensions VALUES (?, ?, ?, ?, ?, ?)', row)

    util.reinitialize_param_table(c2)
    c1.execute('SELECT Name, Value FROM Parameters')
    for name,value in c1:
        c2.execute('UPDATE Parameters SET Value = ? WHERE NAME = ?', (
            value, name))

    c1.close()
    c2.close()

    conn2.commit()

def main():
    parser = create_parser('usage: %s training.sqlite3 test.sqlite [options]' % __file__)
    options, args = parser.parse_args()
    if len(args) != 2:
        parser.error('invalid number of arguments')

    copy_dim(args[0], args[1])

if __name__ == '__main__':
    sys.exit(main())
