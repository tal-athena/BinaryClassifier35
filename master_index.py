import os.path as P

def create_parser():
    from optparse import OptionParser
    p = OptionParser("usage: python %prog train.qlite language [options]")
    p.add_option("-t", "--temporary-dir", type="string", dest="temporary_dir", default=None, help="Specify the directory to use for storing temporary files")
    return p

def main():
    parser = create_parser()
    opts, args = parser.parse_args()
    if len(args) != 2:
        parser.error("invalid number of arguments")
    training_sqlite, process_language = args

    print("Arguments: %s %s" % (training_sqlite, process_language))

    temporary_dir = opts.temporary_dir if opts.temporary_dir else P.dirname(training_csv)
    if not P.isdir(temporary_dir):
        parser.error("error: temporary directory %s does not exist" % temporary_dir)
    print("Temporary dir: %s" % (temporary_dir))

    print("Test message")
    print("OK")

if __name__ == "__main__":
    import sys
    sys.exit(main())

