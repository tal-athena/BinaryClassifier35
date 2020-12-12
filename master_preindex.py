import os.path as P

def create_parser():
    from optparse import OptionParser
    p = OptionParser("usage: python %prog training.csv test.csv language [options]")
    p.add_option("-t", "--temporary-dir", type="string", dest="temporary_dir", default=None, help="Specify the directory to use for storing temporary files")
    p.add_option("-i", "--index-file", type="string", dest="index_file", default=None, help="Specify the path to index file")
    return p

def main():
    parser = create_parser()
    opts, args = parser.parse_args()
    if len(args) != 3:
        parser.error("invalid number of arguments")
    training_csv, test_csv, process_language = args

    print("Arguments: %s %s %s" % (training_csv, test_csv, process_language))

    temporary_dir = opts.temporary_dir if opts.temporary_dir else P.dirname(training_csv)
    if not P.isdir(temporary_dir):
        parser.error("error: temporary directory %s does not exist" % temporary_dir)
    print("Temporary dir: %s" % (temporary_dir))

    index_file = opts.index_file if opts.index_file else None
    if index_file is None:
        parser.error("error: index file not specified")
    if not P.exists(index_file):
        parser.error("error: index file %s does not exist" % index_file)
    print("Index file: %s" % (index_file))
    print("Test message")
    print("OK")


if __name__ == "__main__":
    import sys
    sys.exit(main())
