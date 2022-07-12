from collections import Counter

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

#!pip install ase
from ase.db import connect

#  Databases operations
db = connect("train.db")
#db_test = connect("test.db")

features = []
target = []
for row in db.select():
    features.append([row.natoms, list(row.numbers)])
    target.append(row.data['energy'])

#features_test = []

#for row in db_test.select():
#    features_test.append([row.natoms, list(row.numbers)])

#  Features vectorization

d = DictVectorizer()
features_onehot = d.fit_transform([Counter(x[1]) for x in features])
#features_onehot_test = d.transform([Counter(x[1]) for x in features_test])

#  Dataset splits preparation
target[target.index(min(target))] = sorted(target)[1]
X_train, X_test, y_train, y_test = train_test_split(features_onehot, target, random_state=42)

#  Model selection and training

lr = LinearRegression()
lr.fit(X_train.toarray(), y_train)

print(mean_absolute_error(y_test, lr.predict(X_test.toarray())))

#  Submission preparation

#predictions_test = lr.predict(features_onehot_test.toarray())
#submission_db = pd.DataFrame(columns=["id", "energy"])
#submission_db['id'] = list(range(1, len(predictions_test) + 1))
#submission_db['energy'] = predictions_test
#submission_db.to_csv("sample_submission_lr.csv", index=None)
