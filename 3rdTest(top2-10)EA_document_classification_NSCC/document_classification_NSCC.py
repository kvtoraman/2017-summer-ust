"""
======================================================
Classification of text documents using sparse features
======================================================

This is an example showing how scikit-learn can be used to classify documents
by topics using a bag-of-words approach. This example uses a scipy.sparse
matrix to store the features and demonstrates various classifiers that can
efficiently handle sparse matrices.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
#import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_rcv1

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")

def precision_at_k(eval_set, pred):

	return

				   
def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

##################################################################
### Define train/test/code list files 
#################################################################
#code_list_fn = "./data/KSSC_sample_data_170206_Codelist.dat"
#train_fn = "./data/KSSC_sample_data_170206_Train.dat"
#test_fn = "./data/KSSC_sample_data_170206_Test.dat"

code_list_fn = "NSCC_sample_data_170309_Codelist.dat"
train_fn = "NSCC_sample_data_170309_Train.dat"
test_fn = "NSCC_sample_data_170309_Test.dat"

###############################################################################
# Load some categories from the training set
if opts.all_categories:
    categories = [
        'None'
    ]
else:

	## automatically loading,,,, 
    categories =  [x for x in open(code_list_fn,'r').read().split('\n') if len(x) > 0]
	
if opts.filtered:
    remove = ('headers', 'footers', 'quotes')
else:
    remove = ()


print("Loading NSCC  dataset for categories:")
print(categories if categories else "all")

	
data_train = open(train_fn).readlines()
data_test = open(test_fn).readlines()

data_train_data, data_test_data = [], []
y_train, y_test = [], []
for line in data_train:
	items = line.split('\t')
	if len(items) == 2:
		data_train_data.append(items[1].decode('utf-8', 'ignore'))
		y_train.append(items[0])

for line in data_test:
	items = line.split('\t')
	if len(items) == 2:
		data_test_data.append(items[1].decode('utf-8', 'ignore'))
		y_test.append(items[0])

		
print (len(data_train_data), len(data_test_data))
print('data loaded')

# order of labels in `target_names` can be different from `categories`
target_names = categories #data_train.target_names


# Add Word Embedding (Word Embedding, Topic Embedding, Topic-Event Embedding) Features

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
my_stop_words = [unicode(x.strip(), 'utf-8') for x in open('kor_stop_word.txt','r').read().split('\n')]


#print (my_stop_words)

if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train_data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words=my_stop_words)
    X_train = vectorizer.fit_transform(data_train_data)
	

duration = time() - t0

print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test_data)
duration = time() - t0
print("n_samples: %d, n_features: %d" % X_test.shape)
print()


# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()

if feature_names:
    feature_names = np.asarray(feature_names)


def trim(s):
    #Trim string to fit on terminal (assuming 80-column display)##
    return s if len(s) <= 80 else s[:77] + "..."


###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    ccm = clf.predict(X_test)
    print (ccm)
    #pred_list = clf.predict_proba(X_test)
    #print (clf.classes_, pred_list)
	
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

	# Add F1 socre
    score_acc = metrics.accuracy_score(y_test, pred)
    score_macro_f1 = metrics.f1_score(y_test, pred, average='macro')
    score_micro_f1 = metrics.f1_score(y_test, pred, average='micro')
    print("Precision:   %0.3f" % score_acc)
    print("Macro F1:   %0.3f" % score_macro_f1)
    print("Micro F1:   %0.3f" % score_micro_f1)
    print(metrics.classification_report(y_test, pred, target_names))

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score_acc, train_time, test_time


results = []

for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="auto"), "Ridge Classifier"),
        #(Perceptron(n_iter=50), "Perceptron")
        #(PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        #(KNeighborsClassifier(n_neighbors=10), "kNN"),
        #(RandomForestClassifier(n_estimators=100), "Random forest")
		):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))


for penalty in ["l2"]: #, "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(loss='modified_huber', alpha=.0001, n_iter=50, penalty='elasticnet')))

										   
