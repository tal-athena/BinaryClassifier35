"""
Wrapper for classifier.exe.
"""

#
# TODO: check return codes
#

import sys
import subprocess as sub
import re
import csv
import sqlite3

import os
import os.path as P
import util

from constants import DLIB_CLASSIFIER

def create_parser(usage):
    """Create an object to use for the parsing of command-line arguments."""
    from optparse import OptionParser
    parser = OptionParser(usage)
    parser.add_option("--debug", "-d", dest="debug", default=False, action="store_true", help="Show debug information")
    parser.add_option("--threshold", "-t", dest="threshold", default=0.0, type="float", help="Set the classification threshold")
    return parser

def classifier(database_path, samples_path, classifier_path, threshold=0.0):
    output_dir = P.dirname(samples_path)
    output_csv = P.join(output_dir, "classification-result.csv")
    cmdline = [DLIB_CLASSIFIER, classifier_path, samples_path, output_csv, str(threshold)]
    p = sub.Popen(cmdline, stderr=sub.PIPE, stdout=sub.PIPE)
    stdout, stderr = p.communicate()
    if stdout:
        print (stdout)
    if stderr:
        print(stderr, file=sys.stderr)
        return False

    with open(samples_path+".id") as fin:
        test_ids = fin.read().strip().split("\n")
    scores = []
    with open(output_csv) as fin:
        reader = csv.reader(fin, delimiter=",")
        for score, _ in reader:
            #
            # TODO: why are some scores greater than 1 or less than -1?
            #
            score = int(float(score)*100)
            score = max(-100, min(100, score))
            scores.append(score)
    assert len(scores) == len(test_ids)

    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    for i, doc_id in enumerate(test_ids):
        c.execute("UPDATE Documents SET Score = ? WHERE ED_ENC_NUM = ?", (scores[i], doc_id))
    c.close()
    conn.commit()

    return True

def main():
    parser = create_parser('usage: %s test.sqlite3 test-samples.dat classifier.dat.dlib [options]' % __file__)
    options, args = parser.parse_args()
    if len(args) != 3:
        parser.error('invalid number of arguments')
    classifier(args[0], args[1], args[2], options.threshold)
    return 0

if __name__ == '__main__':
    sys.exit(main())
