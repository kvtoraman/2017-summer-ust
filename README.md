
TODO: Add database(.xls and .dat)

# Code Recommendation for R&D Projects (국가R&D과제의 자동 분류코드 추천)

This is document classifier for National R&D Reports in Korea.(Report are in Korean)
This is my project for [UST Global Research Internship.](https://www.ust.ac.kr/eng.do)
I upgraded the code provided to me by KISTI. KISTI used code provided by [sklearn](http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py).

## Getting Started

### Prerequisites

* [python 2.7](https://www.python.org/download/releases/2.7/)

* [nltk](http://www.nltk.org/install.html)
* [sklearn](http://scikit-learn.org/stable/developers/advanced_installation.html)
* [konlpy](http://konlpy.org/en/v0.4.4/install/)

### How to run

1. Use XLS Parser to parse .xls file into a .dat file
1. (optional) Use english space inserter to fix english part( This was necessary in my case)
1. Run 3rdTest_Final

## Built With

* [Jupyter Notebook](http://jupyter.readthedocs.io/en/latest/install.html)


## Authors

* **Kamil Veli TORAMAN** - *Final work* - [kvtoraman](https://github.com/kvtoraman)

### From sklearn
* **Peter Prettenhofer** <peter.prettenhofer@gmail.com>
* **Olivier Grisel** <olivier.grisel@ensta.org>
* **Mathieu Blondel** <mathieu@mblondel.org>
* **Lars Buitinck**

## Acknowledgments

* KISTI:http://www.kisti.re.kr/eng/
* NTIS:https://www.ntis.go.kr/ThMain.do
* 2017 Summer, UST Global Research Internship, Daejeon, South Korea
