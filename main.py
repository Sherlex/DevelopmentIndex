import itertools

import lr as lr
import matplotlib.pyplot as plt
import np as np
import pandas as pd
import seaborn
import seaborn as sns
from sklearn import svm, clone
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

df = pd.read_csv('C:/Users/elena/PycharmProjects/pythonProject1/Development Index.csv')
df.head(5)
df.info()
sns.heatmap(df.corr(), cmap='viridis', annot=True)
plt.show()
df = df.drop(['Population', 'Area (sq. mi.)', 'Pop. Density '], axis=1)
df.info()
sns.pairplot(df)
plt.show()

plt.figure(figsize=(15, 10))
plt.tight_layout()
seaborn.distplot(df['Development Index'])
plt.show()

Y = df['Development Index']
X = df.drop('Development Index', axis=1)
print(Y.shape)
print(X.shape)

'''
Y_label = label_binarize(Y, classes=[1, 2, 3, 4])
'''

n_samples = X.shape[0]
clf = OneVsRestClassifier(svm.SVC(kernel='linear', C=1, random_state=42))

plt.scatter(X['Infant mortality '], X['Literacy (%)'], c=Y, cmap='winter')
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1, shuffle=True)
'''
cv = ShuffleSplit(n_splits=5, random_state=0)
scores = cross_val_score(clf, X_train, Y_train, cv=cv)
print(scores)
print(np.mean(scores))
'''
clf.fit(X_train, Y_train)
Y_test_pred = clf.predict(X_test)

accuracy = clf.score(X_test, Y_test)
print(accuracy)
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_test_pred})
df1 = df.head(25)
print(df1)
df1.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cnf_matrix = confusion_matrix(Y_test, Y_test_pred)
plt.figure(figsize=(10, 8))
plot_confusion_matrix(cnf_matrix, classes=['1', '2', '3', '4'],
                      title='Confusion matrix')
plt.show()

report = classification_report(Y_test, Y_test_pred, target_names=['1', '2', '3', '4'])
print(report)
