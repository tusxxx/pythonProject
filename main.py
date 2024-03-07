import pandas as pd
from sklearn import tree

data = pd.read_csv('train.csv')
data = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].dropna()
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

x = data[['Pclass', 'Fare', 'Age', 'Sex']]
y = data['Survived']
# print(y)
clf = tree.DecisionTreeClassifier(random_state=241)
clf.fit(x, y)

importances = pd.Series(clf.feature_importances_, index=list(x))
sorted_importances = importances.sort_values()
print(sorted_importances[-2:])
