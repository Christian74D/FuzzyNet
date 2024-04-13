A hybrid classification model built by using Pytorch Neural Networks and Fuzzy TOPSIS head for correlative inference between different hormones and variants of Thyroid Disease.


Video demo https://youtu.be/3hBURpzFnv4

Dataset used https://www.kaggle.com/datasets/emmanuelfwerr/thyroid-disease-data

Required modules:
pandas
numpy
matplotlib
sklearn
torch
tqdm

Usage:
1) Install required modules using pip
2) Modify constants.py to vary no of desicion makers or other constants
3) Run train.py to train the model
4) Run test.py to test the model

Files:
1) constants.py
has various constants for deciding parameters of the model, like no of decision makers, learning rate, no of epochs,..

2) data.py
loads the pre-cleaned csv file 'thyroid_cleaned_num.csv' as pandas df, and splits into train and test loaders using sklearn

3) decision_maker.py
object class for individual decision-makers in fuzzy topsis

4) fuzzy_topsis.py
object class for implementation of fuzzy topsis methodology

5) model.py
FuzzyNet model which integrates fuzzy_topsis for output class prediction

6) plots.py
helper functions to display/save plots

7) train.py
trains the FuzzyNet model

8) test.py
evaluates the saved FuzzyNet model's accuracy and displays the confusion matrix

9) inference.py
plots the input-output curves of individual networks for various hormones, helping in visualizing the correlation in hormone levels and the classification of the disease  

![image](https://github.com/Christian74D/FuzzyNet/assets/112863270/3818b131-d4fa-4b6e-8567-dcc42dea02d7)

