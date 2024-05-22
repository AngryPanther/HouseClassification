from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier(random_state=42, max_depth=6)

clf = clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))

print(clf.score(X_test, y_test))