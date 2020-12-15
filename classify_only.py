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
from index_04_09 import index
from mrmrnew_05 import mrmr
from svmvec_06_10 import svmvec
from svm_learn_07 import learn
from copy_dim_08 import copy_dim
from svm_classify_11 import classify
from prep_sqlitedb_00 import prep_sqlite, CSTRING
import spacy


def create_parser():
    from optparse import OptionParser
    p = OptionParser("usage: python %prog training.sqlite3 test.sqlite3 [options]")
    p.add_option("-t", "--temporary-dir", type="string", dest="temporary_dir", default=None, help="Specify the directory to use for storing temporary files")
    p.add_option("-c", "--classifier-dir", type="string", dest="classifier_dir", default=None, help="Specify the directory of the classifier")
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

    temporary_dir = opts.temporary_dir if opts.temporary_dir else P.dirname(test_sqlite3)
    if not P.isdir(temporary_dir):
        parser.error("error: temporary directory %s does not exist" % temporary_dir)
    else:
        print("Test directory: %s" % (temporary_dir))
        test_sqlite3 = P.join(temporary_dir, test_sqlite3)

    classifier_dir = opts.classifier_dir if opts.classifier_dir else P.dirname(training_sqlite3)
    if not P.isdir(classifier_dir):
        parser.error("error: classifier directory %s does not exist" % classifier_dir)
    else:
        print("Classifier directory: %s" % (classifier_dir))
        training_sqlite3 = P.join(classifier_dir, training_sqlite3)

    def log(message):
        print ("[%s] %s" % (datetime.datetime.now().isoformat(), message))



    #
    # Check all tables are valid
    #
    if not P.isfile(training_sqlite3):
        print ("File not exist, ", training_sqlite3);
        return;

    if not P.isfile(test_sqlite3):
        print ("File not exist, ", test_sqlite3);
        return;

    conn = sqlite3.connect(training_sqlite3)

    c = conn.cursor();
    
    c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='Dimensions' ''')
    if c.fetchone()[0] !=1 :
        print ('Dimensions table does not exist.')
        return
    
    c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='Parameters' ''')
    if c.fetchone()[0]!=1 :
        print('Parameters table does not exist.')
        return;
    
    try:
        c.execute("""SELECT DimensionId, Term, PartOfSpeech, Exclude, IDF, MRMR
                FROM Dimensions WHERE Exclude = 0""")

        flag = False
        for row in c:
            flag = True
            break;
        if flag == False:
            print("Dimensions table is empty")
            return;

        c.execute('SELECT Name, Value FROM Parameters')
        
        flag = False
        for row in c:
            flag = True
            break;
        if flag == False:
            print("Parameters table is empty")
            return;
    except sqlite3.Error as error:
        print("Table schema error:", error)
        return;
    finally:
        if (conn):
            conn.close()            

    nlp = spacy.load(process_language)

    #
    # Test section
    #
    log("Preparing SQLite test database")    
    prep_sqlite(test_sqlite3)

    log("Copying dimensions from training database to test database")
    copy_dim(training_sqlite3, test_sqlite3)

    log("Indexing test database")
    
    index(temporary_dir, test_sqlite3, nlp, True)    
    
    log("Outputting test samples to temporary data file")
    test_samples = P.join(temporary_dir, "test-samples.dat")
    svmvec(test_sqlite3, test_samples)

    log("Classifying test samples")
    classifier = P.join(classifier_dir, "classifier.svm")
    classify(test_sqlite3, test_samples, classifier, False, temporary_dir)

if __name__ == "__main__":
    import sys
    sys.exit(main())


