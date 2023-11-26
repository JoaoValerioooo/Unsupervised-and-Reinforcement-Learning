Unsupervised and Reinforcement Learning
Practical Work 1: An extended version of the k-means method for overlapping clustering
João Valério
joao.agostinho@estudiantat.upc.edu
09/05/2023

Requirements:
beautifulsoup4==4.12.2
numpy==1.24.2
pandas==1.5.3
scikit_learn==1.2.2
scipy==1.10.1
tabulate==0.9.0
scikit-fuzzy==0.4.2

Software:
PyCharm CE

Programming Language:
Python - version 3.8.8

Execution:
To execute the code through the terminal the following steps should be taken:
pip install beautifulsoup4
pip install numpy
pip install pandas
pip install scikit_learn
pip install scipy
pip install tabulate
pip install scikit-fuzzy
python3 /PATH_WHERE_THE_FILE_main.py_IS_INSERTED
Ex: /Users/joaovalerio/Documents/"MAI UPC"/"2 Semester"/URL/W1/source/main.py
/PATH_WHERE_THE_FOLDER_DATA_IS_INSERTED
Ex: /Users/joaovalerio/Documents/MAI UPC/2 Semester/URL/W1

PW1-URL-2223-JOAOVALERIO Folder structure:

- Documentation
---- W1_Report.pdf: Report of the present project.
---- C_ICPR_08.pdf: Original paper implemented in the present project.

- Source
---- Reuters.py: contains the Reuters class, which consists of the preprocessing of the Reuters-21578 dataset characterised in Chapters 2.a and 2.b.
---- Genetic.py: contains the Genetic class, which consists of the preprocessing of the Yeast dataset characterised in Chapters 2.c and 2.d.
---- Models.py: contain the implementation of the MODELS class with the KM, FKM and OKM algorithms.
---- main.py: the main .py file, where the preprocessing and clustering stages are executed.

- Data
---- reuters21578: folder with the Reuters-21578 dataset.
---- Yeast: folder with the Yeast dataset.
---- Results.txt: the results achieved and exposed on the report.

- README.txt