"""
Include/exclude dimensions from the index using automatic mRMR feature 
selection is applied.
"""

import sqlite3
import sys
import nltk
from nltk.corpus import stopwords

import util
from calc_dim_01 import match_exclude_regex

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

def mrmr(c, path):
    """
    Perform automatic mRMR feature selection using the specified cursor.
    Changes are persisted to the database using the cursor.
    """
    params = util.get_params(c, path)
    #
    # mRMR feature selection
    #
    include_dim = set()
    exclude_dim = set()
    all_dim = util.get_dimensions(c, 0)

    c.execute('SELECT COUNT(ED_ENC_NUM) FROM Documents')
    num_total_docs = int(c.fetchone()[0])
    c.execute('SELECT COUNT(ED_ENC_NUM) FROM Documents WHERE Score > 0')
    num_positive_docs = int(c.fetchone()[0])
    c.execute('SELECT COUNT(ED_ENC_NUM) FROM Documents WHERE Score < 0')
    num_negative_docs = int(c.fetchone()[0])
    
    #
    # The part below is ported from filterFeatures() of reference.py
    #
    cu = params['C_UPPERCUTOFF'] * num_total_docs
    ccp = params['C_CLASSCUTOFF'] * num_positive_docs
    ccm = params['C_CLASSCUTOFF'] * num_negative_docs
    lcp = params['C_LOWERCUTOFF'] * num_positive_docs
    lcm = params['C_LOWERCUTOFF'] * num_negative_docs

    #
    # The original script didn't have any comments, so here's my guess of what
    # individual variables represent.
    #
    # cu        Upper cut-off.  If a feature occurs in more than cu documents,
    #           then it should be excluded.
    # ccp       Upper class cut-off for positive documents.
    # lcp       Lower class cut-off for positive documents.
    #           If the frequency of a feature within positive documents 
    #           falls within this interval, then it should be excluded.
    # ccm       Upper class cut-off for negative documents.
    # lcm       Lower class cut-off for negative documents.
    #           If the frequency of a feature within negative documents 
    #           falls within this interval, then it should be excluded.
    #

    for (dim_id, _, _) in all_dim:
        text_count, plus_count, minus_count = 0, 0, 0
        c.execute("""SELECT Score
                FROM Documents INNER JOIN DocumentsToDimensions
                ON Documents.ED_ENC_NUM = DocumentsToDimensions.ED_ENC_NUM
                WHERE DimensionId = ?""", (dim_id,))
        for score in c:
            text_count += 1
            if score > 0:
                plus_count += 1
            elif score < 0:
                minus_count += 1

        if params['USE_UPPERCUTS'] and text_count > cu:
            exclude_dim.add(dim_id)
            if dim_id in include_dim:
                include_dim.remove(dim_id)
        elif params['USE_CLASSCUTS'] and minus_count > ccm and plus_count > ccp : 
            exclude_dim.add(dim_id)
            if dim_id in include_dim:
                include_dim.remove(dim_id)
        elif params['USE_LOWERCUTS'] and minus_count < lcm and plus_count < lcp : 
            exclude_dim.add(dim_id)
            if dim_id in include_dim:
                include_dim.remove(dim_id)
        else:
            if dim_id in exclude_dim:
                exclude_dim.remove(dim_id)
            include_dim.add(dim_id)
    #
    # end of ported code.
    #
    print ('mRMR enabled:', len(include_dim), 'disabled:', len(exclude_dim))

    assert not include_dim.intersection(exclude_dim)
    for dim in include_dim:
        c.execute(
                'UPDATE Dimensions SET Exclude = 0 WHERE DimensionId = ?',
                (dim,))
    for dim in exclude_dim:
        c.execute(
                'UPDATE Dimensions SET Exclude = 1 WHERE DimensionId = ?',
                (dim,))

def main():
	try:
		parser = create_parser('usage: %s file.sqlite3 [options]' % __file__)
		options, args = parser.parse_args()
		if not len(args):
			parser.error('invalid number of arguments')
		conn = sqlite3.connect(args[0])
		c = conn.cursor()
		mrmr(c)

		conn.commit()
		c.close()
		return 0
	except:
		print ("Unexpected error:", sys.exc_info()[0])
		return 0

if __name__ == '__main__':
    sys.exit(main())
