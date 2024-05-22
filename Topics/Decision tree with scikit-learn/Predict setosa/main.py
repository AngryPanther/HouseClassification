from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(random_state=42)
clf = clf.fit(X, y)

df = pd.read_csv("data\\dataset\\input.txt")

prediction = clf.predict(df)

result = 0
for i in prediction:
    if i == 0:
        result += 1

print(result)
