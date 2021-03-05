import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

titantic = pd.read_csv('C:/Users/User/Desktop/python/titanic/train.csv')


tmp = []
for each in titantic['Sex']:
    if each =='female':
        tmp.append(1)
    elif each == 'male':
        tmp.append(0)
    else :
        tmp.append('np.nan')
print(titantic.info())        
titantic['Sex'] = tmp

titantic['Survived'] = titantic['Survived'].astype('int')
titantic['Pclass']  = titantic['Pclass'].astype('float')
titantic['Sex'] = titantic['Sex'].astype('float') 
titantic['SibSp'] = titantic['SibSp'].astype('float')
titantic['Parch'] = titantic['Parch'].astype('float')
titantic['Fare'] = titantic['Fare'].astype('float')

titantic = titantic[titantic['Age'].notnull()]
titantic = titantic[titantic['SibSp'].notnull()]
titantic = titantic[titantic['Parch'].notnull()]
titantic = titantic[titantic['Fare'].notnull()]

train_data = titantic[['Pclass','Sex','Age','SibSp','Parch','Fare']]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_data, titantic['Survived'],test_size=0.2, random_state=777)


from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train,y_train)

print('Score :{}',format(tree_clf.score(X_train,y_train)))

from sklearn.tree import export_graphviz

export_graphviz(
        tree_clf,
        out_file="titanic.dot",
        feature_names=['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare'],
        class_names=['Unsurvived','Survived'],
        rounded=True,
        filled=True
    )

import graphviz
with open("titanic.dot") as f:
    dot_graph = f.read()
dot = graphviz.Source(dot_graph)
dot.format = 'png'
dot.render(filename='titanic_tree', directory='images/decision_trees', cleanup=True)
dot