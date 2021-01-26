import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


df = pd.read_csv('penguins.csv')
df.fillna(df.mean(), inplace=True)

sex = df['sex']
species = df['species']
df['_class'] = sex + ' ' + species

target = df['_class']

df['ordered_species'] = pd.Categorical(df.species,ordered=True,categories=['Adelie', 'Chinstrap', 'Gentoo']).codes
df['ordered_island'] = pd.Categorical(df.island,ordered=True,categories=['Torgersen', 'Biscoe', 'Dream']).codes
df['ordered_class'] = pd.Categorical(df._class,ordered=True,categories=['Male Adelie', 'Female Adelie', 'Male Chinstrap', 'Female Chinstrap','Male Gentoo','Female Gentoo']).codes + 1

df['is_female'] = df['sex'] == 'Female'
df.fillna(df.mean(), inplace=True)
df.drop(['sex', 'species', 'island','_class'], axis=1, inplace=True)

sns.pairplot(df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'ordered_class']], hue='ordered_class', height=1.5)
plt.show()

