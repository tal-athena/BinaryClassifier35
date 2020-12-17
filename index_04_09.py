"""Indexes documents from the Documents table."""

import os.path as P
import csv
import json
import sqlite3
import sys
import util
import math
import time
import spacy
from timeit import default_timer as timer

from math import log10
from calc_dim_01 import process_document_spacy
import os.path as P
class Batch(object):
    def __init__(self, i, document_ids):
        self.number = i
        self.document_ids = document_ids

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
    parser.add_option(
            '--limit',
            '-l',
            dest='limit',
            type='int',
            default=0,
            help='Limit the number of documents to process')
    parser.add_option(
            '--subprocesses',
            '-N',
            dest='subprocesses',
            type='int',
            default=1,
            help='Specify the number of subprocesses to use')
    parser.add_option(
            '--batch-size',
            '-b',
            dest='batch_size',
            type='int',
            default=100,
            help='Specify the number of documents to process per batch')
    return parser

def remove_unnecessary_dimensions(filename):
    conn = sqlite3.connect(filename)
    cur = conn.cursor()
    cur.execute('SELECT DimensionId FROM Dimensions')
    dimension_ids = [ d[0] for d in cur.fetchall() ]

    for dim in dimension_ids:
        cur.execute("SELECT count(*) FROM DocumentsToDimensions WHERE DimensionId=%d" % dim)
        cnt = cur.fetchone()[0]
        if cnt == 0:
            cur.execute('DELETE FROM Dimensions WHERE DimensionId=%d'% dim)
    conn.commit()
    conn.close()

def add_external_dimensions(temporary_dir, filename, is_test):

    print('add_external_dimensions(): processing %s' % filename)

    parameter_file = P.join(temporary_dir, "parameters.json")

    if P.exists(parameter_file) == False: return

    f = open(parameter_file)
    data = json.load(f)

    if is_test:
        if "pathToTestParametersCSV" in data: csv_file_name = data["pathToTestParametersCSV"]
        else: return
    else:
        if "pathToTrainParametersCSV" in data: csv_file_name = data["pathToTrainParametersCSV"]
        else: return
    
    conn = sqlite3.connect(filename)
    cur = conn.cursor()
    if is_test == False:
        cur.execute('select MAX(DimensionId) FROM Dimensions')
        new_dimension_id = cur.fetchone()[0] + 1

    with open(csv_file_name, newline='') as csvfile:
        dim_dict = {}
        
        reader = csv.DictReader(csvfile, delimiter=';')    
        field_names = reader.fieldnames
        if is_test == True:
            for col in field_names:
                if col == 'ED_ENC_NUM': continue
                cur.execute('SELECT DimensionId FROM Dimensions WHERE Term="' + col + '"')
                result = cur.fetchone()
            if result != None: dim_dict[col] = result[0]
        else:
            for col in field_names:
                if col == 'ED_ENC_NUM': continue
                cur.execute('INSERT INTO Dimensions VALUES (?, ?, ?, 0, 0, 0)',
                   (new_dimension_id, col,'RegEx'))
                dim_dict[col] = new_dimension_id
                new_dimension_id = new_dimension_id + 1
            conn.commit()
            
        for row in reader:       
            for col in field_names:
                if col != 'ED_ENC_NUM' and int(row[col]) > 0 and col in dim_dict:
                    cur.execute('INSERT INTO DocumentsToDimensions (DimensionId, ED_ENC_NUM, Count) VALUES (?, ?, ?)',
                       (dim_dict[col], row['ED_ENC_NUM'], row[col]))
            
            conn.commit()
    
    conn.close()
    print('add_external_dimensions(): %s finished' % filename)

def index(temporary_dir, filename, nlp, is_test = False):
    """
    Perform indexing.  Each document is stemmed, and then the non-excluded
    dimensions are counted for that document.  The result is put into the
    DocumentsToDimensions table.
    """
    conn = sqlite3.connect(filename)
    c = conn.cursor()

    doc_to_dim_exist = util.check_if_tables_exist( c, [ 'DocumentsToDimensions' ])
    if doc_to_dim_exist:
        print ('Using pre-calculated DocumentsToDimensions.')
        c.close()
        return

    # Create the table
    util.init_dims_to_docs(c)

    params = util.get_params(c, filename)
    stemmer = params['stemmer']
    print ('index(): stemmer: %s' % stemmer)


    all_dim = util.get_dimensions(c, 0)
    assert all_dim, "You must calculate dimensions prior to indexing."

    all_include = util.get_all_include_regex(c)

    c.execute('SELECT COUNT(ED_ENC_NUM) FROM Documents')
    num_total_docs = int(c.fetchone()[0])

    c.execute('DELETE FROM DocumentsToDimensions')

    c.execute("SELECT COUNT(*) FROM Dimensions WHERE PartOfSpeech = 'bigram'")
    nBigrams = int(c.fetchone()[0])
    print ('Number of bigrams: ', nBigrams)
    do_bigrams = nBigrams > 0

    c.execute("SELECT COUNT(*) FROM Dimensions WHERE PartOfSpeech = 'trigram'")
    nTrigrams = int(c.fetchone()[0])
    print ('Number of trigrams: ', nTrigrams)
    do_trigrams = nTrigrams > 0

    #
    # If the POS column contains "unigram", then it means we didn't perform POS tagging when calculating dimensions.
    #
    c.execute("SELECT COUNT(*) FROM Dimensions WHERE PartOfSpeech = 'unigram'")
    pos_tag = int(c.fetchone()[0]) == 0

    cmd = 'SELECT ED_ENC_NUM FROM Documents'
    # if options.limit:
    #    cmd += ' LIMIT %d' % options.limit
    #    num_total_docs = min(options.limit, num_total_docs)

    #
    # TODO: why is fetchmany not working?
    #
    #document_ids = c.execute(cmd).fetchmany()
    document_ids = []
    for row in c.execute(cmd):
        document_ids.append(row[0])
    print ("fetched %d document ids" % len(document_ids))
    
    #
    # Terminate the SQL connection so that the subprocesses can use it.
    #
    conn.commit()
    conn.close()

    #
    # https://docs.python.org/2/library/array.html#module-array
    #
    
    main_process(nlp, document_ids, filename, stemmer, all_include, pos_tag, do_bigrams, do_trigrams, all_dim)  
    

    conn = sqlite3.connect(filename)
    c = conn.cursor()
    util.add_indexes_for_dimensions_to_docs(c)

    conn.commit()
    c.close()

    add_external_dimensions(temporary_dir, filename, is_test)
    remove_unnecessary_dimensions(filename)

    conn = sqlite3.connect(filename)
    c = conn.cursor()

    start = timer()
    util.log ("Starting IDF calculation")

    for dim_id, _, _ in all_dim:
        c.execute("""SELECT COUNT(DimensionId)
                FROM DocumentsToDimensions
                WHERE DimensionId = ?""", (dim_id,))
        freq = int(c.fetchone()[0])
        idf = log10(num_total_docs/(1+freq))
        c.execute(
                'UPDATE Dimensions SET IDF = ? WHERE DimensionId = ?',
                (idf, dim_id))

    #
    # Save and exit.
    #
    conn.commit()
    c.close()

    end = timer()
    util.log ('End IDF calculation elapsed time: {:f} seconds'.format(end - start))

def main_process(nlp, document_ids, fpath, stemmer, all_include, pos_tag, do_bigrams, do_trigrams, all_dim):
    """Read document numbers from input_queue, read the actual documents from the database, process_document them, write back to the database."""
    documents = {}
    value_list = []
    conn = cursor = None
    while True:
        delay = 1
        try:
            conn = sqlite3.connect(fpath)
            cursor = conn.cursor()
            for doc_number in document_ids:
                documents[doc_number] = cursor.execute("SELECT NOTE_TEXT FROM Documents WHERE ED_ENC_NUM = ?", (doc_number,)).fetchone()[0]
            break
        except sqlite3.OperationalError:
            print ("index_document(): database is locked, trying again in %ds" %delay)
            time.sleep(delay)
            delay += 1
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    value_list = []

    for ordinal, doc_number in enumerate(documents):
        proc = process_document_spacy(nlp, documents[doc_number], stemmer, all_include, pos_tag, do_bigrams, do_trigrams)
        for dimension, term, pos in all_dim:
            if pos == 'bigram':
                count = proc['bigrams_counter'][term]
            elif pos == 'trigram':
                count = proc['trigrams_counter'][term]
            elif pos == 'regex':
                count = proc['inclusions_counter'][term]
            else:
                count = proc['stemmed_counter'][(pos, term)]
            if not count:
                continue
            value_list.append((dimension, doc_number, count))

    conn = cursor = None
    
    while True:
        delay = 1
        try:
            conn = sqlite3.connect(fpath)
            cursor = conn.cursor()
            cursor.executemany('INSERT INTO DocumentsToDimensions VALUES ( ?, ?, ? )', value_list)
            conn.commit()
            break
        except sqlite3.OperationalError:
            if (delay > 10):
                break
            print ("index_document(): database is locked, trying again in %ds" % delay)
            time.sleep(delay)
            delay += 1
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    print ("index_document():", len(documents))


def main():
    parser = create_parser('usage: %s file.sqlite3 [options]' % __file__)
    options, args = parser.parse_args()
    if not len(args):
        parser.error('invalid number of arguments')
    index(args[0], options)
    return 0

if __name__ == '__main__':
    sys.exit(main())
