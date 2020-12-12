"""
Extracts the Documents table from an SQlite3 file into an MS Access MDB file.
Ignores all other tables.
The contents of the destination file will be completely overwritten.

Requires a working PYODBC with an installed MS Access Driver (probably
Windows only).
"""
import sys
import os.path as P
current_dir, _ = P.split(P.abspath(__file__))
sys.path.append(P.join(current_dir, '..'))
import util

if len(sys.argv) != 3:
    print ('usage: %s file.sqlite3 empty.mdb' % __file__)
    sys.exit(1)

import pyodbc
cstring = 'DRIVER={Microsoft Access Driver (*.mdb)};DBQ=%s' % sys.argv[2]
try:
    conn_out = pyodbc.connect(cstring)
except pyodbc.Error:
    print >> sys.stderr, '"%s" is not an existing Access database file' % sys.argv[2]
    sys.exit(1)
c_out = conn_out.cursor()

import sqlite3
conn_in = sqlite3.connect(sys.argv[1])
c_in = conn_in.cursor()

util.create_documents_table(c_out)
c_in.execute('SELECT ED_ENC_NUM, NOTE_TEXT, Score from Documents')
for doc_id, text, score in c_in:
    c_out.execute('INSERT INTO Documents VALUES (?,?,?)', (doc_id, text, score))

c_in.close()

conn_out.commit()
c_out.close()
sys.exit(0)
