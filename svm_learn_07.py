"""
Wrapper for svm_learn.exe.
"""

import sqlite3
import sys
import subprocess as sub
import re

import os
import os.path as P
import util

from constants import SVM_LEARN

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

def learn(path, training, svm_classifier):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    param = util.get_params(c, path)
    if param['SVM_LEARN']:
        options = param['SVM_LEARN'].split(' ')
    else:
        options = []

    cmdline = [SVM_LEARN] + options + [training, svm_classifier]
    p = sub.Popen(cmdline, stderr=sub.PIPE, stdout=sub.PIPE)
    stdout, stderr = p.communicate()
    if stderr:
        print(stderr, file=sys.stderr)
        return

    c.close()

def main():
    parser = create_parser('usage: %s training.sqlite3 svm_training svm_classifier [options]' % __file__)
    options, args = parser.parse_args()
    if len(args) != 3:
        parser.error('invalid number of arguments')
    learn(args[0], args[1], args[2])
    return 0

if __name__ == '__main__':
    sys.exit(main())
