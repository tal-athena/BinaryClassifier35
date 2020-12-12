#
# Implements the training and classification workflow in a single script.
#
import os.path as P
import os
import pyodbc
import sqlite3
import datetime

from calc_dim_01 import calc_dim
from exclude_02 import exclude_stopwords, exclude_shorter_than, exclude_non_alpha_partial
from prune_03 import prune
from index_04_09 import index
from mrmrnew_05 import mrmr
from svmvec_06_10 import svmvec
from svm_learn_07 import learn
from copy_dim_08 import copy_dim
from svm_classify_11 import classify
from mdb2sqlite import mdb2sqlite, CSTRING

class IndexingOptions(object):
    def __init__(self):
        self.limit = 0
        self.batch_size = 100
        self.subprocesses = 2

def create_parser():
    from optparse import OptionParser
    p = OptionParser("usage: python %prog training.mdb test.mdb [options]")
    p.add_option("-t", "--temporary-dir", type="string", dest="temporary_dir", default=None, help="Specify the directory to use for storing temporary files")
    return p

def copy_scores(src_sqlite, dest_mdb):
    conn = sqlite3.connect(src_sqlite)
    c = conn.cursor()
    scores = {}
    for (doc_id, score) in c.execute("SELECT ED_ENC_NUM, Score FROM Documents"):
        scores[doc_id] = score
    c.close()
    conn.close()

    conn = pyodbc.connect(CSTRING % dest_mdb)
    c = conn.cursor()
    for doc_id in scores:
        c.execute("UPDATE Documents SET Score = ? WHERE ED_ENC_NUM = ?", (scores[doc_id], doc_id))
    c.close()
    conn.commit()
    conn.close()

def main():
    parser = create_parser()
    opts, args = parser.parse_args()
    if len(args) != 2:
        parser.error("invalid number of arguments")

    training_mdb, test_mdb = args

    temporary_dir = opts.temporary_dir if opts.temporary_dir else P.dirname(training_mdb)
    if not P.isdir(temporary_dir):
        parser.error("error: temporary directory %s does not exist" % temporary_dir)

    def log(message):
        print ("[%s] %s" % (datetime.datetime.now().isoformat(), message))

    #
    # Training section
    #
    log("Converting training database from MS Access to SQLite")
    training_sqlite = P.join(temporary_dir, P.splitext(P.basename(training_mdb))[0]+".sqlite3")
    mdb2sqlite(training_mdb, training_sqlite, zero_score=False)

    log("Calculating dimensions")
    calc_dim(training_sqlite)
    log("Excluding dimensions")
    exclude_stopwords(training_sqlite)
    exclude_non_alpha_partial(training_sqlite)
    exclude_shorter_than(training_sqlite, 3)
    log("Pruning excluded dimensions")
    prune(training_sqlite)
    log("Indexing training database")
    index(training_sqlite, IndexingOptions())
    log("Running mRMR algorithm to select features")
    mrmr(training_sqlite, temporary_dir)
    log("Pruning excluded dimensions (again)")
    prune(training_sqlite)

    log("Outputting training samples to temporary data file")
    training_samples = P.join(temporary_dir, "training-samples.dat")
    svmvec(training_sqlite, training_samples)

    log("Training classifier")
    classifier = P.join(temporary_dir, "classifier.svm")
    learn(training_sqlite, training_samples, classifier)

    #
    # Test section
    #
    log("Converting test database from MS Access to SQLite")
    test_sqlite = P.join(temporary_dir, P.splitext(P.basename(test_mdb))[0]+".sqlite3")
    mdb2sqlite(test_mdb, test_sqlite, zero_score=True)

    log("Copying dimensions from training database to test database")
    copy_dim(training_sqlite, test_sqlite)

    log("Indexing test database")
    index(test_sqlite, IndexingOptions())
    
    log("Outputting test samples to temporary data file")
    test_samples = P.join(temporary_dir, "test-samples.dat")
    svmvec(test_sqlite, test_samples)

    log("Classifying test samples")
    classify(test_sqlite, test_samples, classifier, False, temporary_dir)

    #
    # This won't work on OS/X since the ODBC driver is read-only :(
    # 
    log("Copying scores to test database (MS Access)")
    copy_scores(test_sqlite, test_mdb)

if __name__ == "__main__":
    import sys
    sys.exit(main())
