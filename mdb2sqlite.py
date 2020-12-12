"""
Extracts the Documents table from an MS Access MDB file into a sqlite3 file.
The contents of the destination file will be overwritten.

Requires a working PYODBC with an installed MS Access Driver (probably
Windows only).
"""
import sys
import os.path as P
import util
import pyodbc
import sqlite3

from constants import CSTRING

def mdb2sqlite(src_mdb, dest_sqlite, zero_score=False):
    cstring =  CSTRING % src_mdb
    print (cstring)
    conn_in = pyodbc.connect(cstring)
    c_in = conn_in.cursor()

    conn_out = sqlite3.connect(dest_sqlite)
    c_out = conn_out.cursor()

    fout = open(dest_sqlite, 'w')
    fout.close()

    util.create_documents_table(c_out)
    util.reinitialize_regex_tables(c_out)
    util.reinitialize_param_table(c_out)

    counter = 0
    for tbl in ('Documents', 'Documents_pos', 'Documents_neg'):
        try:
            c_in.execute('SELECT ED_ENC_NUM, NOTE_TEXT, Score FROM %s' % tbl)
        except (pyodbc.ProgrammingError, pyodbc.Error) as error:
            #
            # The table doesn't exist.  Quietly try the next table name.
            #
            continue
        for doc_id, text, score in c_in:
            if zero_score:
                score = 0

            c_out.execute('insert into Documents values ( ?, ?, ? )', (doc_id, text, score))
            counter += 1
    print ("Inserted %d documents" % counter)

    c_in.close()

    conn_out.commit()
    c_out.close()

def create_parser(usage):
    """Create an object to use for the parsing of command-line arguments."""
    from optparse import OptionParser
    parser = OptionParser(usage)
    parser.add_option('--zero-score', '-z', dest='zero_score', default=False, action='store_true', help='Set Score column to zero')
    return parser

def main():
    parser = create_parser('usage: %s file.sqlite3 [options]' % __file__)
    options, args = parser.parse_args()
    if len(args) != 2:
        parser.error('invalid number of arguments')
    mdb2sqlite(args[0], args[1], options.zero_score)
    sys.exit(0)

if __name__ == "__main__":
    main()
