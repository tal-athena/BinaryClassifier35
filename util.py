"""
Utility functions.  
"""
import os.path as P
import sqlite3
import re
import sre_constants
import json
import datetime
from timeit import default_timer as timer

def log(message):
        print ("[%s] %s" % (datetime.datetime.now().isoformat(), message))

def create_documents_table(c):
    c.execute("""CREATE TABLE Documents ( 
            ED_ENC_NUM DOUBLE PRIMARY KEY,
            NOTE_TEXT TEXT, 
            Score INTEGER
        )""")

def reinitialize_regex_tables(c):
    drop_tables(c, [ 'InclusionRegex', 'ExclusionRegex' ])
    c.execute("""CREATE TABLE InclusionRegex (
            InclusionId INTEGER PRIMARY KEY,
            Regex TEXT
        )""")
    c.execute("""CREATE TABLE ExclusionRegex (
            ExclusionId INTEGER PRIMARY KEY,
            Regex TEXT,
            PartOfSpeechFilter TEXT
        )""")

DEFAULT_PARAMS = {
        'USE_BINARIZED_TDF' : True,
        'USE_LOWERCUTS' : True,
        'USE_UPPERCUTS' : True,
        'USE_CLASSCUTS' : True,
        'C_LOWERCUTOFF' : 0.025,
        'C_UPPERCUTOFF' : 0.95,
        'C_CLASSCUTOFF' : 0.66,
        'C_BINARIZE' : 0.001,
        'stemmer' : 'porter',
        'bigrams' : False,
        'trigrams' : False,
        'SVM_LEARN' : '-v 2 -c 0.01 -j 0.5 -t 0 -p 0.03 -m 512',
        'CLASSIFY_CLIP': 3.0,
        'MRMR' : '-n 250 -t 1'
        }

def reinitialize_param_table(c):
    drop_tables(c, ['Parameters'])
    c.execute('CREATE TABLE Parameters (Name TEXT PRIMARY KEY, Value TEXT)')
    for name,value in list(DEFAULT_PARAMS.items()):
        c.execute('INSERT INTO Parameters VALUES (?,?)', (name, str(value)))

def get_params(c, path):
    c.execute('SELECT Name, Value FROM Parameters')
    params = {}
    for name,value in c:
        default_type =  type(DEFAULT_PARAMS[name])
        if default_type is bool:
            #
            # boolean constructor appears to be broken?
            #
            params[name] = True if value.lower() == 'true' else False
        else:
            params[name] = default_type(value)

    #
    # If any data is missing in the Parameters table, pull in the defaults.
    #
    for default in DEFAULT_PARAMS:
        if default not in params:
            params[default] = DEFAULT_PARAMS[default]

    temporary_dir = P.dirname(path)
    param_path = P.join(temporary_dir, 'parameters.json')
    if not P.isfile(param_path):
        print ("Parameters file: %s isn't found. Using default parameters." % (param_path))
        return params
    input_params = json.load(open(param_path))
    nDim = input_params['numberOfDimensions']
    if isinstance(nDim, int):
        params['MRMR'] = '-n ' + str(nDim) + ' -t 1'

    includeBigrams = input_params['includeBigrams']
    if isinstance(includeBigrams, (bool)):
        params['bigrams'] = includeBigrams

    includeTrigrams = input_params['includeTrigrams']
    if isinstance(includeTrigrams, (bool)):
        params['trigrams'] = includeTrigrams
    
    return params

def get_all_include_regex(c):
    """
    Read inclusion regexes from the database and compile them.
    Invalid regexes will fail compilation and not be included in the result.
    """
    c.execute('SELECT InclusionId, Regex FROM InclusionRegex')
    fetched = c.fetchall()
    result = []
    for iid, regex in fetched:
        try:
            regex = re.compile(regex, re.IGNORECASE)
            result.append(regex)
        except sre_constants.error as error:
            print ('skipping invalid regex (%d): `%s\' (%s)' % (iid,regex,error))
    return result

def get_all_exclude_regex(c):
    """
    Read exclusion regexes from the database and compile them.
    Invalid regexes will fail compilation and not be included in the result.
    """
    c.execute("""SELECT ExclusionId, Regex, PartOfSpeechFilter
            FROM ExclusionRegex""")
    fetched = [ (r[0], r[1], r[2].split(' ')) for r in c.fetchall() ]
    result = []
    for eid, regex, pos_list in fetched:
        try:
            regex = re.compile(regex, re.IGNORECASE)
            result.append((regex, pos_list))
        except sre_constants.error as error:
            print ('skipping invalid regex (%d): `%s\' (%s)' % (eid,regex,error))
    return result

def drop_tables(c, tables):
    """
    Check that the specified tables exist, and for those that do, drop them.
    """
    all_tables = get_all_tables(c)
    for t in tables:
        if t in all_tables:
            c.execute('DROP TABLE %s' % t)

def get_all_tables(c):
    """Get a list of all the tables in the database."""
    all_tables = []
    c.execute('SELECT name FROM SQLITE_MASTER WHERE type = "table"')
    for tbl in c:
        all_tables.append(tbl[0])
    return all_tables

def output_dim_table(c):
    c.execute('SELECT Term, IDF, MRMR FROM Dimensions WHERE Exclude = 0 ORDER BY MRMR DESC')
    print('------------------------------------------------')
    print('----------------- Dimensions -------------------')
    print ('%30s  %30s  %30s' % ('Term', 'IDF', 'MRMR'))
    for dim in c:
        print ('%30s  %30s  %30s' % (dim[0], dim[1], dim[2]))
    print('-------------- End of Dimensions ---------------')
    print('------------------------------------------------')


def get_dimensions(c, exclude=None):
    """
    Get dimensions from the Dimensions table.

    If the optional argument exclude is not specified, fetches ALL the 
    dimensions.  If it's zero, fetches included dimensions only.
    If it's 1, fetches excluded dimensions only.
    """
    if exclude is None:
        c.execute("""SELECT DimensionId, Term, PartOfSpeech, Exclude
                FROM Dimensions""")
    else:
        c.execute("""SELECT DimensionId, Term, PartOfSpeech 
                FROM Dimensions where Exclude = ?""", (exclude,))
    return c.fetchall()


def check_if_tables_exist(c, tables):
    all_tables = get_all_tables(c)
    for t in tables:
        if t not in all_tables:
            return False
    return True;


def reset_dimensions(c, exclude=False):
    """
    Set the Exclude field of the specified dimension to the specified value.
    """
    cmd = 'UPDATE Dimensions SET Exclude = ?'
    c.execute(cmd, (exclude))

def reset_note_text(c):
    cmd = 'UPDATE Documents SET NOTE_TEXT = ?'
    c.execute(cmd, (''))


def init_dims_to_docs(c):
    c.execute("""CREATE TABLE DocumentsToDimensions (
            DimensionId LONG, 
            ED_ENC_NUM DOUBLE,
            Count INTEGER
        )""")

def add_indexes_to_documents(c):
    drop_indexes_for_documents(c)
    createSecondaryIndex = "CREATE INDEX idx_Documents_ED_ENC_NUM ON Documents(ED_ENC_NUM)"
    c.execute(createSecondaryIndex)

def drop_indexes_for_documents(c):
    try:
        createSecondaryIndex = "DROP INDEX idx_Documents_ED_ENC_NUM"
        c.execute(createSecondaryIndex)
    except:
        print ('') # do nothing


def add_indexes_for_dimensions(c):
    start = timer()
    drop_indexes_for_dimensions(c)
    createSecondaryIndex = "CREATE INDEX idx_Dimensions_DimensionId ON Dimensions(DimensionId)"
    c.execute(createSecondaryIndex)
    end = timer()
    print ('Add indices to Dimensions table elapsed time: {:f} seconds'.format(end - start))

def add_indexes_for_dimensions_to_docs(c):
    start = timer()
    drop_indexes_for_dimensions_to_docs(c)
    createSecondaryIndex1 = "CREATE INDEX idx_DocumentsToDimensions_DimensionId ON DocumentsToDimensions(DimensionId, ED_ENC_NUM)"
    c.execute(createSecondaryIndex1)
    createSecondaryIndex2 = "CREATE INDEX idx_DocumentsToDimensions_ED_ENC_NUM ON DocumentsToDimensions(ED_ENC_NUM)"
    c.execute(createSecondaryIndex2)
    end = timer()
    print ('Add indices to DimensionsToDocs table elapsed time: {:f} seconds'.format(end - start))

def drop_indexes_for_dimensions(c):
    try:
        createSecondaryIndex = "DROP INDEX idx_Dimensions_DimensionId"
        c.execute(createSecondaryIndex)
    except:
        print ('') # do nothing

def drop_indexes_for_dimensions_to_docs(c):
    try:
        createSecondaryIndex1 = "DROP INDEX idx_DocumentsToDimensions_DimensionId"
        c.execute(createSecondaryIndex1)
        createSecondaryIndex2 = "DROP INDEX idx_DocumentsToDimensions_ED_ENC_NUM"
        c.execute(createSecondaryIndex2);
    except:
        print ('') # do nothing
