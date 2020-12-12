"""Creates an sqlite3 database using the Shakespeare corpus of NLTK."""
import nltk
import sqlite3
import sys
import os
import os.path as P
current_dir, _ = P.split(P.abspath(__file__))
sys.path.append(P.join(current_dir, '..'))
import util
import random

def scrub_non_ascii(text):
    result = []
    for i in text:
        if ord(i) < 128:
            result.append(i)
    return ''.join(result)

def texts_to_sqlite(fname, texts):
    """
    Write the specified texts to a Documents table of the specified file.
    Each string in the list texts will become a separate document.
    The Documents table in the destination file will be completely
    overwritten.
    """
    conn = sqlite3.connect(fname)
    c = conn.cursor()

    util.reinitialize_regex_tables(c)
    util.reinitialize_param_table(c)

    util.drop_tables(c, [ 'Documents' ])
    util.create_documents_table(c)

    #
    # Shuffle the document IDs around.
    # Keeping them as ordinals is unrealistic.
    #
    document_ids = range(len(texts))
    random.shuffle(document_ids)
    for i,t in enumerate(texts):
        c.execute('INSERT INTO Documents VALUES ( ?, ?, ?)', 
                (document_ids[i], scrub_non_ascii(t), 0))

    conn.commit()
    c.close()

def split_into_blocks(words, bsize):
    """Split a list of words into blocks at most bsize words each."""
    if len(words) < bsize:
        return [ ' '.join(words) ]

    blocks = []
    for i in range(bsize, len(words), bsize):
        blocks.append(' '.join(words[i-bsize:i]))
    return blocks

def main():
    from nltk.corpus import shakespeare
    from nltk.corpus import gutenberg
    from nltk.corpus import brown
    from nltk.corpus import inaugural
    from nltk.corpus import webtext
    #
    # This is a bit of a hack, but I can't seem to find a better way to
    # obtain the plaintext.
    #
    texts = []
    for corpus in (shakespeare, gutenberg, brown, inaugural, webtext):
        for f in corpus.fileids():
            print (f)
            texts += split_into_blocks(corpus.words(f), 1000)

    texts_to_sqlite('shakespeare.sqlite3', texts)

if __name__ == '__main__':
    main()
