"""
Include/exclude dimensions from the index using automatic mRMR feature 
selection is applied.
"""

import sqlite3
import sys
import nltk
from nltk.corpus import stopwords
import re

import util
import subprocess as sub

import os.path as P

from constants import MRMR
from util import output_dim_table
from util import DEFAULT_PARAMS

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import time

from shutil import copyfile

MRMR_EXE_FEATURE_LIMIT = 10000

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

    return parser


def reduce_mrmr_input(temp_dir) :

    print("reduce_mrmr_input....")

    copyfile(P.join(temp_dir, "mrmr-in.csv"), P.join(temp_dir, "mrmr-in-full.csv"))

    col_names = pd.read_csv(P.join(temp_dir, "mrmr-in.csv"), nrows=0).columns
    types_dict = {'Class': float}
    types_dict.update({col: int for col in col_names if col not in types_dict})

    train = pd.read_csv(P.join(temp_dir, "mrmr-in.csv"), dtype=types_dict)
    
    df = train.drop("Class", axis =1)
    
    if len(df.columns) <= 8000:
        print("mrmr no need to reduced features (<8000)")
        return len(df.columns);

    start = time.process_time()       

    model = RandomForestRegressor(random_state=1, max_depth=100)    
    model.fit(df, train.Class)

    print("model.fit taken time", time.process_time() - start)

    features = df.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[-8000:]  # top 8000 features
    cols = ['Class']
    for x in indices:
        cols.append(features[x])

    train.set_index("Class")
    train.to_csv(P.join(temp_dir, "mrmr-in.csv"), mode = 'w', columns=cols, index=False)
    print("mrmr reduced features to 8000")
    return 8000;

def mrmr(path, temp_dir):
    conn = sqlite3.connect(path)
    c = conn.cursor()

    c.execute('SELECT DimensionId FROM Dimensions')
    dimension_ids = [ d[0] for d in c.fetchall() ]

    mrmr_tmp = P.join(temp_dir, "mrmr-in.csv")
    fout = open(mrmr_tmp, 'w')
    fout.write(','.join([ 'Class' ] + list(map(str, dimension_ids))) + '\n')
    
    #
    # Output the CSV file for the mRMR utility.
    #
    c.execute('SELECT ED_ENC_NUM, Score FROM Documents')
    num_doc = 0
    for doc_id, score in c:
        #
        # Feature selection can only take place when we have labelled samples.
        #
        assert score in (-100, 100)
        c_inner = conn.cursor()
        nonzero = {}
        #
        # TODO: ignore disabled dimensions?
        #
        c_inner.execute("""SELECT DimensionId, Count
                FROM DocumentsToDimensions where ED_ENC_NUM = ?""",
                (doc_id,))
        for dim_id, count in c_inner:
            nonzero[dim_id] = count
        values = [ str(score/100) ]
        for dim in dimension_ids:
            if dim in nonzero:
                values.append(nonzero[dim])
            else:
                values.append(0)
        fout.write(','.join(map(str, values)) + '\n')
        num_doc += 1

    fout.close()

    # select top 8000    
    n_features = reduce_mrmr_input(temp_dir)

    #
    # Run the mRMR utility.
    #
    params = util.get_params(c, path)

    cmd = [ MRMR, '-i', mrmr_tmp, '-s', str(num_doc),
            '-v', str(n_features) ] + params['MRMR'].split(' ')
    print('command line:', ' '.join(cmd))
    p = sub.Popen(cmd, bufsize=1, stdout=sub.PIPE, stderr=sub.STDOUT)

    #
    # This blocks until the underlying process completes, so can appear 
    # unresponsive.  Don't do this.
    #
    # stdout, stderr = p.communicate()

    #
    # Parse the output, enable/disable the required features.
    # There's a warning about buffers filling up and blocking the process if
    # things are done this way 
    # (http://docs.python.org/library/subprocess.html#subprocess.Popen.kill).
    # However, in our case, there isn't THAT much data to worry about -- it's
    # more important to output it as it's coming in so it looks like the
    # application is actually doing something.
    #
    result = {}
    regex = re.compile('(\\d+) \t (\\d+) \t (\\d+) \t (\\d+\\.\\d+)')

    #
    # Argh, stdout is still being buffered...
    # TODO: try https://bitbucket.org/geertj/winpexpect/wiki/Home
    #
    while True:
        line = p.stdout.readline().decode('utf-8')
        if not line:
            break
        print(line, end=' ')
        match = regex.search(line)
        if not match:
            continue
        order, fea, name, score = match.groups()
        result[int(order)] = (int(fea), int(name), float(score))
    print()

    selected = sorted([ (result[k][1], result[k][2]) for k in result ])
    c.execute('UPDATE Dimensions SET Exclude = 1')
    for dim in selected:
        c.execute(
                'UPDATE Dimensions SET Exclude = 0, MRMR = ? WHERE DimensionId = ?', 
                (dim[1], dim[0]))
    
    conn.commit()
    output_dim_table(c)
    c.close()

def main():
    parser = create_parser('usage: %s file.sqlite3 [options]' % __file__)
    options, args = parser.parse_args()
    if not len(args):
        parser.error('invalid number of arguments')
    mrmr(args[0], P.dirname(args[0]))
    return 0

if __name__ == '__main__':
    sys.exit(main())
