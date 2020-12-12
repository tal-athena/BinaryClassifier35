"""
Create some tables in the sqlite database
"""
import sys
import os.path as P
import util
import sqlite3

from constants import CSTRING

def prep_sqlite(dest_sqlite):
  
    conn_out = sqlite3.connect(dest_sqlite)
    c_out = conn_out.cursor()

    util.reinitialize_regex_tables(c_out)
    util.reinitialize_param_table(c_out)

    conn_out.commit()
    c_out.close()

def create_parser(usage):
    """Create an object to use for the parsing of command-line arguments."""
    from optparse import OptionParser
    parser = OptionParser(usage)
    return parser

def main():
    parser = create_parser('usage: %s file.sqlite3 [options]' % __file__)
    options, args = parser.parse_args()
    if len(args) != 1:
        parser.error('invalid number of arguments')
    prep_sqlite(args[0])
    sys.exit(0)

if __name__ == "__main__":
    main()
