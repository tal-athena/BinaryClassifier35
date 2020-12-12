"""
Wrapper for trainer.exe.
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

from constants import DLIB_TRAINER

def create_parser(usage):
    """Create an object to use for the parsing of command-line arguments."""
    from optparse import OptionParser
    parser = OptionParser(usage)
    parser.add_option("--debug", "-d", dest="debug", default=False, action="store_true", help="Show debug information")
    return parser

def parameter_selection(training):
    """Returns the best parameter C for the specified training data file."""
    output_file = P.join(P.dirname(training), "parameter-selection.csv")
    cmdline = [DLIB_TRAINER, training, output_file]
    p = sub.Popen(cmdline, stderr=sub.PIPE, stdout=sub.PIPE)
    stdout, stderr = p.communicate()
    if stdout:
        print (stdout)
    if stderr:
        print (stderr, file=sys.stderr)
        print ("Aborting.", file=sys.stderr)
        return -1

    best_c = -1
    best_f1 = 0
    with open(output_file) as fin:
        reader = csv.reader(fin, delimiter=",")
        for row in reader:
            c, p, r = map(float, row)
            f1 = 2*p*r/(p+r)
            if f1 > best_f1:
                best_c = c
                best_f1 = f1

    return best_c

def main():
    parser = create_parser('usage: %s svm_training [options]' % __file__)
    options, args = parser.parse_args()
    if len(args) != 1:
        parser.error('invalid number of arguments')
    best_c = parameter_selection(args[0])
    print (best_c)
    return 0

if __name__ == '__main__':
    sys.exit(main())
