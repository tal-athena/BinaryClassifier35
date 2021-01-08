#
# Implements the training and classification workflow in a single script.
#
import os.path as P
import os
import sqlite3
import datetime

from calc_dim_01 import calc_dim
from exclude_02 import exclude_stopwords_spacy, exclude_shorter_than, exclude_non_alpha_partial, exclude_unigrams_shorter_than, exclude_ngrams_shorter_than
from prune_03 import prune
from index_04_09 import index, index_with_pre
from mrmrnew_05 import mrmr
from svmvec_06_10 import svmvec
from svm_learn_07 import learn
from copy_dim_08 import copy_dim, copy_doc_to_dim
from svm_classify_11 import classify
from prep_sqlitedb_00 import prep_sqlite, CSTRING
import spacy
import json
from timeit import default_timer as timer

def create_parser():
    from optparse import OptionParser
    p = OptionParser("usage: python %prog training.sqlite3 test.sqlite3 [options]")
    p.add_option("-t", "--temporary-dir", type="string", dest="temporary_dir", default=None, help="Specify the directory to use for storing temporary files")
    return p

def main():
    parser = create_parser()
    opts, args = parser.parse_args()
    if len(args) != 3:
        parser.error("invalid number of arguments")

    training_sqlite3, test_sqlite3, process_language = args

    print("Arguments: %s %s %s" % (training_sqlite3, test_sqlite3, process_language))

    #cwd = os.getcwd()
    #print "Working directory: %s" % (cwd)

    temporary_dir = opts.temporary_dir if opts.temporary_dir else P.dirname(training_sqlite3)
    if not P.isdir(temporary_dir):
        parser.error("error: temporary directory %s does not exist" % temporary_dir)

    def log(message):
        print ("[%s] %s" % (datetime.datetime.now().isoformat(), message))

    nlp = spacy.load(process_language)

    #
    # Training section
    #
    log("Preparing SQLite training database")
    training_sqlite = P.join(temporary_dir, training_sqlite3)
    prep_sqlite(training_sqlite)

    parameter_file = P.join(temporary_dir, "parameters.json")

    use_index_file = False
    index_sqlite = ''
    if P.exists(parameter_file) == True:
        f = open(parameter_file)
        data = json.load(f)
        if 'indexDir' in data:
            use_index_file = True
            index_sqlite = P.join(data['indexDir'], 'index.sqlite3')

    if use_index_file == True:
        start = timer()
        copy_dim(index_sqlite, training_sqlite)
        copy_doc_to_dim(index_sqlite, training_sqlite)
        end = timer()
        print ('Copy Dimenions and DimensionsToDocuments table from indexed db to train db elapsed time: {:f} seconds'.format(end - start))
        index_with_pre(temporary_dir, training_sqlite, index_sqlite, nlp, False)
    else:

        log("Calculating dimensions")
        calc_dim(nlp, training_sqlite, 0, False)
        log("Excluding dimensions")

        #if process_language == 'en':
        #    exclude_stopwords(training_sqlite)        
        #else :
        #    exclude_stopwords_spacy(training_sqlite, process_language)

        exclude_stopwords_spacy(nlp, training_sqlite, process_language)

        exclude_non_alpha_partial(training_sqlite)
        exclude_unigrams_shorter_than(training_sqlite, 3)
        exclude_ngrams_shorter_than(training_sqlite, 1)
        log("Pruning excluded dimensions")
        prune(training_sqlite)
        log("Indexing training database")

        index(temporary_dir, training_sqlite, nlp)    

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
    log("Preparing SQLite test database")
    test_sqlite = P.join(temporary_dir, test_sqlite3)
    prep_sqlite(test_sqlite)

    log("Copying dimensions from training database to test database")
    copy_dim(training_sqlite, test_sqlite)

    log("Indexing test database")
    if use_index_file == True:
        start = timer()
        copy_doc_to_dim(index_sqlite, test_sqlite)
        end = timer()
        print ('Copy Dimenions and DimensionsToDocuments table from indexed db to test db elapsed time: {:f} seconds'.format(end - start))
        index_with_pre(temporary_dir, test_sqlite, index_sqlite, nlp, True)
    else :
        index(temporary_dir, test_sqlite, nlp, True)    
    
    log("Outputting test samples to temporary data file")
    test_samples = P.join(temporary_dir, "test-samples.dat")
    svmvec(test_sqlite, test_samples)

    log("Classifying test samples")
    classify(test_sqlite, test_samples, classifier, False, temporary_dir)

if __name__ == "__main__":
    import sys
    sys.exit(main())

