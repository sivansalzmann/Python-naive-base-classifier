import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def bayes_plot(penguins,className, model="gnb", spread=30):
    penguins.dropna()
    colors = 'seismic'
    col1 = penguins.columns[0]
    col2 = penguins.columns[1]
    target = penguins.columns[2]
    y = penguins[target]
    X = penguins.drop(target, axis=1)
    print(y)
    print(X)

    # Task 1.2,2.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    clf = GaussianNB()

    if (model != "gnb"):
        clf = DecisionTreeClassifier(max_depth=model)
    clf = clf.fit(X_train, y_train)

    prob = len(clf.classes_) == 2

    y_pred = clf.predict(X_test)

    # Task 1.5,2.5
    print(metrics.classification_report(y_test, y_pred))

    hueorder = clf.classes_

    def numify(val):
        return np.where(clf.classes_ == val)[0]

    Y = y.apply(numify)
    x_min, x_max = X.loc[:, col1].min() - 1, X.loc[:, col1].max() + 1
    y_min, y_max = X.loc[:, col2].min() - 1, X.loc[:, col2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),np.arange(y_min, y_max, 0.2))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    if prob:
        Z = Z[:, 1]-Z[:, 0]
    else:
        colors = "Set1"
        Z = np.argmax(Z, axis=1)

    # Task 1.3,2.3
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=colors, alpha=0.5)
    plt.colorbar()
    if not prob:
        plt.clim(0, len(clf.classes_)+3)

    # Task 1.4,2.4
    y_pred2 = clf.predict(X)
    ypred = pd.Series(y_pred2, name="prediction")
    predicted1 = pd.concat([X.reset_index(), y.reset_index(), ypred], axis=1)
    predicted2 = predicted1[predicted1[className] != predicted1.prediction]
    col11 = predicted2.columns[1]
    col21 = predicted2.columns[2]
    target1 = predicted2.columns[4]
    sns.scatterplot(data=predicted2[::spread], x=col11, y=col21, hue=target1, hue_order=hueorder, palette=colors)

    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.show()



if __name__ == "__main__":
    # Task 1.1
    penguins = pd.read_csv("penguins.csv")
    penguins = penguins.dropna()

    graph = sns.pairplot(penguins, hue='species', height=1.5)
    plt.show()

    y = penguins['species']
    x = penguins.drop(['species', 'sex', 'island',
                       'body_mass_g', 'bill_depth_mm'], axis=1)

    bayes_plot(pd.concat([x, y], axis=1),'species', spread=1)


    # Task 2.1
    penguins["class"] = penguins["sex"]+" "+penguins["species"]
    graph = sns.pairplot(penguins, hue='class', height=1.5)
    plt.show()

    y2 = penguins['class']
    x2 = penguins.drop(['species', 'sex', 'island','class',
                        'body_mass_g', 'flipper_length_mm'], axis=1)

    bayes_plot(pd.concat([x2, y2], axis=1),'class', spread=1)