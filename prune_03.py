"""
Removes excluded dimensions from Vectors and Dimensions.

Note that this operation is not undo-able.
"""

import sqlite3
import sys
import nltk
import util

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

def prune(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()

    doc_to_dim_exist = util.check_if_tables_exist( c, [ 'DocumentsToDimensions' ])
    if doc_to_dim_exist:
        c.execute("""DELETE FROM DocumentsToDimensions WHERE DimensionId IN (
                SELECT DimensionId FROM Dimensions WHERE Exclude = 1)""")

    c.execute('DELETE FROM Dimensions WHERE Exclude = 1')

    #
    # Save and exit.
    #
    conn.commit()
    c.close()

def main():
    parser = create_parser('usage: %s file.sqlite3 [options]' % __file__)
    options, args = parser.parse_args()
    if not len(args):
        parser.error('invalid number of arguments')
    prune(args[0])
    return 0

if __name__ == '__main__':
    sys.exit(main())
