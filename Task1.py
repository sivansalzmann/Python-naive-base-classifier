import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

df = pd.read_csv('penguins.csv')

df['ordered_species'] = pd.Categorical(df.species,ordered=True,categories=['Adelie', 'Chinstrap', 'Gentoo']).codes
df['ordered_island'] = pd.Categorical(df.island,ordered=True,categories=['Torgersen', 'Biscoe', 'Dream']).codes
df['is_female'] = df['sex'] == 'Female'
df.fillna(df.mean(), inplace=True)
df.drop(['sex', 'species', 'island'], axis=1, inplace=True)

sns.pairplot(df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'ordered_species']], hue='ordered_species', height=1.5)
plt.show()

# 1.1
# After this graph, we cna appreciate that body_mass_g vs bill_length_mm gives a good split.
# Bill_length_mm vs bill_depth_mm seems interesting too.

X = df[['body_mass_g', 'bill_length_mm']]
Y = df['ordered_species']

#1.2
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,random_state=1)

model = GaussianNB()
model.fit(Xtrain,Ytrain)
y_model = model.predict(Xtest)

ypred = pd.Series(y_model,name="prediction") #make prediciton coulmn
predicted = pd.concat([Xtest.reset_index(),Ytest.reset_index()],axis=1)
print("bill_depth_mm vs bill_length_mm accuracy: ", metrics.accuracy_score(Ytest, y_model))

# We see that bill_depth_mm vs bill_length_mm has better accuracy. Lets plot it

def bayes_plot(df,model="gnb",spread=30):
    df.dropna()
    colors = 'seismic'
    col1 = df.columns[0]
    col2 = df.columns[1]
    target = df.columns[2]
    sns.scatterplot(data=df, x=col1, y=col2,hue=target)
    plt.show()
    y = df[target]  # Target variable
    X = df.drop(target, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)  # 80% training and 20% test

    clf = GaussianNB()
    if (model != "gnb"):
        clf = DecisionTreeClassifier(max_depth=model)
    clf = clf.fit(X_train, y_train)

    # Train Classifer

    prob = len(clf.classes_) == 2

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    #1.5
    print(metrics.classification_report(y_test, y_pred))

    hueorder = clf.classes_
    def numify(val):
        return np.where(clf.classes_ == val)[0]

    Y = y.apply(numify)
    x_min, x_max = X.loc[:, col1].min() - 1, X.loc[:, col1].max() + 1
    y_min, y_max = X.loc[:, col2].min() - 1, X.loc[:, col2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    if prob:

        Z = Z[:,1]-Z[:,0]
    else:
        colors = "Set1"
        Z = np.argmax(Z, axis=1)

    #1.3
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=colors, alpha=0.5)
    plt.colorbar()
    if not prob:
        plt.clim(0,len(clf.classes_)+3)
    #1.4
    sns.scatterplot(data=df[::spread], x=col1, y=col2, hue=target, hue_order=hueorder,palette=colors)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.show()

bayes_plot(pd.concat([X,Y],axis=1),spread=1)






