"""Read a CSV file into an sqlite3 table."""

import sqlite3
import sys
import csv

import sys
sys.path.append('..')
import utils

reader = csv.reader(open(sys.argv[1], 'rt'), delimiter=',', quotechar='|')
rows = [ r for r in reader ]
for r in rows:
    for c in r:
        print (repr(c[:10]), end=' ')
    print

headers = rows[0][:3]
data_types = [ '%s text' % h for h in headers ]

fout = open(sys.argv[2], 'w')
fout.close()

conn = sqlite3.connect(sys.argv[2])
c = conn.cursor()

try:
    c.execute('DROP TABLE Documents')
except sqlite3.OperationalError:
    pass

util.create_documents_table()

for row in rows[1:]:
    row = row[:3]
    if len(row) != 3:
        continue
    doc_id, text, score = row[:3]
    c.execute(
            'INSERT INTO Documents VALUES ( ?, ?, ? )',
            (int(doc_id), text, score))

conn.commit()
c.close()
