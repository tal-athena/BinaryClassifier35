import os.path as P
import os
import sqlite3
import datetime

from calc_dim_01 import calc_dim
from exclude_02 import exclude_stopwords_spacy, exclude_shorter_than, exclude_non_alpha_partial, exclude_unigrams_shorter_than, exclude_ngrams_shorter_than
from prune_03 import prune
from index_04_09 import index_without_additional_dimension
from mrmrnew_05 import mrmr
from svmvec_06_10 import svmvec
from svm_learn_07 import learn
from copy_dim_08 import copy_dim
from svm_classify_11 import classify
from prep_sqlitedb_00 import prep_sqlite, CSTRING
import spacy
import shutil

def create_parser():
    from optparse import OptionParser
    p = OptionParser("usage: python %prog train.qlite language [options]")
    p.add_option("-t", "--temporary-dir", type="string", dest="temporary_dir", default=None, help="Specify the directory to use for storing temporary files")
    return p

def main():
    parser = create_parser()
    opts, args = parser.parse_args()
    if len(args) != 2:
        parser.error("invalid number of arguments")
    training_sqlite, process_language = args

    print("Arguments: %s %s" % (training_sqlite, process_language))

    temporary_dir = opts.temporary_dir if opts.temporary_dir else P.dirname(training_csv)
    if not P.isdir(temporary_dir):
        parser.error("error: temporary directory %s does not exist" % temporary_dir)
    print("Temporary dir: %s" % (temporary_dir))

    nlp = spacy.load(process_language)
    
    index_sqlite = P.join(temporary_dir, 'index.sqlite3')
    training_sqlite = P.join(temporary_dir, training_sqlite)
    shutil.copy(training_sqlite, index_sqlite)

    log("Preparing SQLite training database")
    prep_sqlite(index_sqlite)

    log("Calculating dimensions")
    calc_dim(nlp, index_sqlite, 0, False)
    log("Excluding dimensions")

    exclude_stopwords_spacy(nlp, index_sqlite, process_language)

    exclude_non_alpha_partial(index_sqlite)
    exclude_unigrams_shorter_than(index_sqlite, 3)
    exclude_ngrams_shorter_than(index_sqlite, 1)
    log("Pruning excluded dimensions")
    prune(index_sqlite)
    log("Indexing training database")

    index_without_additional_dimension(temporary_dir, index_sqlite, nlp)

def log(message):
        print ("[%s] %s" % (datetime.datetime.now().isoformat(), message))

if __name__ == "__main__":
    import sys
    sys.exit(main())

