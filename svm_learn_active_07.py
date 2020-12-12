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

def trainer(training, svm_classifier, c_value):
    """Trains an SVM classifier from the specified training data file and saves it to the specified filename.
    Returns True on success, false otherwise."""
    cmdline = [DLIB_TRAINER, training, svm_classifier, str(c_value)]
    p = sub.Popen(cmdline, stderr=sub.PIPE, stdout=sub.PIPE)
    stdout, stderr = p.communicate()
    if stdout:
        print (stdout)
    if stderr:
        print(stderr, file=sys.stderr)
        return False
    return True

def main():
    parser = create_parser('usage: %s svm_training svm_classifier c_value [options]' % __file__)
    options, args = parser.parse_args()
    if len(args) != 3:
        parser.error('invalid number of arguments')
    cvalue = float(args[2])
    trainer(args[0], args[1], cvalue)
    return 0

if __name__ == '__main__':
    sys.exit(main())
