"""
Wrapper for ranker.exe.
"""

#
# TODO: check return codes
#

import sys
import subprocess as sub
import re
import csv

import os
import os.path as P
import util

from constants import DLIB_RANKER

def create_parser(usage):
    """Create an object to use for the parsing of command-line arguments."""
    from optparse import OptionParser
    parser = OptionParser(usage)
    parser.add_option("--debug", "-d", dest="debug", default=False, action="store_true", help="Show debug information")
    return parser

def ranker(training, test, c_value):
    """Trains an SVM classifier from the specified training data file and saves it to the specified filename.
    Returns True on success, false otherwise."""
    output_dir = P.dirname(training)
    indices_fname = P.join(output_dir, "ranker-indices.txt")
    cmdline = [DLIB_RANKER, training, test, str(c_value), indices_fname]
    p = sub.Popen(cmdline, stderr=sub.PIPE, stdout=sub.PIPE)
    stdout, stderr = p.communicate()
    if stdout:
        print (stdout)
    if stderr:
        print (stderr, file=sys.stderr)
        return False
    with open(indices_fname) as fin:
        indices = map(int, fin.read().strip().split("\n"))
    with open(test+".id") as fin:
        test_ids = fin.read().strip().split("\n")
    assert len(test_ids) == len(indices)
    with open(P.join(output_dir, "ranks.dat"), "w") as fout:
        for i in indices:
            print >> fout, test_ids[i]

    return True

def main():
    parser = create_parser('usage: %s training test c_value [options]' % __file__)
    options, args = parser.parse_args()
    if len(args) != 3:
        parser.error('invalid number of arguments')
    cvalue = float(args[2])
    ranker(args[0], args[1], cvalue)
    return 0

if __name__ == '__main__':
    sys.exit(main())
