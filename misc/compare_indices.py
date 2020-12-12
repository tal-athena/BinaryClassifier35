import sqlite3
import sys

c1 = sqlite3.connect(sys.argv[1]).cursor()
c2 = sqlite3.connect(sys.argv[1]).cursor()

c1.execute("SELECT * FROM DocumentsToDimensions")
for dim_id, doc_id, count1 in c1:
    print (dim_id)
    c2.execute("""SELECT Count
        FROM DocumentsToDimensions 
        WHERE DimensionId = ? AND ED_ENC_NUM = ?""", (dim_id, doc_id))
    count2 = c2.fetchone()[0]
    assert count1 == count2
