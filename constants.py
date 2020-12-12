import os.path as P
import sys

CURRENT_DIR = P.split(P.abspath(__file__))[0]

if sys.platform == "darwin":
    #
    # Running on my Mac/Linux
    #
    MRMR = P.join(CURRENT_DIR, 'mrmr', 'mrmr')
    SVM_LEARN = "/svmlight/svm_learn"
    SVM_CLASSIFY = "/svmlight/svm_classify"
    CSTRING = 'DRIVER={Actual Access};DBQ=%s'
    DLIB_TRAINER = P.join(CURRENT_DIR, "active-learning", "bin", "trainer.out")
    DLIB_RANKER = P.join(CURRENT_DIR, "active-learning", "bin", "ranker.out")
    DLIB_CLASSIFIER = P.join(CURRENT_DIR, "active-learning", "bin", "classifier.out")
else:
    #
    # Production environment (Windows)
    #
    SVM_LEARN = P.join(CURRENT_DIR, 'svmlight', 'svm_learn.exe')
    SVM_CLASSIFY = P.join(CURRENT_DIR, 'svmlight', 'svm_classify.exe')
    MRMR = P.join(CURRENT_DIR, 'mrmr', 'mrmr_win32.exe')
    CSTRING = 'DRIVER={Microsoft Access Driver (*.mdb)};DBQ=%s'
    DLIB_TRAINER = P.join(CURRENT_DIR, "active-learning", "bin", "trainer.exe")
    DLIB_RANKER = P.join(CURRENT_DIR, "active-learning", "bin", "ranker.exe")
    DLIB_CLASSIFIER = P.join(CURRENT_DIR, "active-learning", "bin", "classifier.exe")
