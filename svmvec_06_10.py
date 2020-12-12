"""Print vector representation as a data file for svmlight."""

import sqlite3
import sys
import nltk
import util
import collections

from math import log10

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

def svmvec(path, output_filename):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c_inner = conn.cursor()
    c_inner2 = conn.cursor()

    params = util.get_params(c, path)

    c.execute('SELECT COUNT(ED_ENC_NUM) FROM Documents')
    num_total_docs = int(c.fetchone()[0])

    c.execute('select ED_ENC_NUM, Score from Documents')
    i = 1
    with open(output_filename, 'w') as fout_samples:
        with open(output_filename+".id", 'w') as fout_ids:
            for doc_id, score in c:
                if i % 100 == 0:
                    print(('svmvec(): processing document %s (%d/%d)' % (str(doc_id), i, num_total_docs)))
                c_inner.execute("""SELECT DocumentsToDimensions.DimensionId, Count
                        FROM DocumentsToDimensions INNER JOIN Dimensions
                        ON DocumentsToDimensions.DimensionId = Dimensions.DimensionId
                        WHERE DocumentsToDimensions.ED_ENC_NUM = ?
                        AND Dimensions.Exclude = 0
                        AND Count > 0""", (doc_id,))
                if score == None:
                    score = 0
                elif score > 100:
                    score = 100
                elif score < -100:
                    score = -100
                assert -100 <= score <= 100
                print('%d' % (score/100), end=' ', file=fout_samples)
                print(doc_id, file=fout_ids)
                for dim_id, count in c_inner:
                    c_inner2.execute("""SELECT IDF FROM Dimensions
                        WHERE DimensionId = ?""", (dim_id,))
                    idf = float(c_inner2.fetchone()[0])

                    #
                    # The SELECT statement above protects us from zero count.
                    #
                    tfidf = 1 + log10(count)*idf
                    if params['USE_BINARIZED_TDF']:
                        tfidf = 1 if tfidf > float(params['C_BINARIZE']) else 0
                        print('%d:%d' % (dim_id, tfidf), end=' ', file=fout_samples)
                    else:
                        print('%d:%d' % (dim_id, tfidf), end=' ', file=fout_samples)
                print(file=fout_samples)
                i += 1

    c_inner.close()
    c_inner2.close()
    c.close()
    conn.close()

def main():
    parser = create_parser('usage: %s file.sqlite3 output.dat [options]' % __file__)
    options, args = parser.parse_args()
    if len(args) != 2:
        parser.error('invalid number of arguments')
    svmvec(args[0], args[1])
    return 0

if __name__ == '__main__':
    sys.exit(main())
