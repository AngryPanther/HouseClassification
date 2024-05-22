import os
import requests
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
import functions as fc

if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")

    df = pd.read_csv('../Data/house_class.csv')
    cols1 = ['Zip_area', 'Zip_loc', "Room"]
    cols2 = ['Zip_area', 'Room', "Zip_loc"]

    # DataFrame X is used to predict the price in DataFrame y
    X, y = df, df["Price"]

    # Train/Test Split between the two DataFrames
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1,
                                                        stratify=X['Zip_loc'].values)

    # Begin OneHotEncoder


    enc = OneHotEncoder(drop='first')
    enc.fit(X[['Zip_area', 'Zip_loc', 'Room']])

    X_train_transformed = pd.DataFrame(enc.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
                                       index=X_train.index, dtype=int).add_prefix('enc')

    X_test_transformed = pd.DataFrame(enc.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
                                      index=X_test.index, dtype=int).add_prefix('enc')

    X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)
    X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)

    clf = fc.dtc()
    clf = clf.fit(X_train_final, y_train)

    train_score = clf.score(X_test_final, y_test)

    # End OneHotEncoder

    # Classification Report
    from sklearn.metrics import classification_report
    report = classification_report(clf.predict(X_test_final), y_test, output_dict=True)
    print("OneHotEncoder:" + str(round(report['macro avg']['f1-score'],2)))

    # End Classification Report

    # Begin Ordinal Encoder


    enc = OrdinalEncoder()
    enc.fit(X[['Zip_area', 'Zip_loc', 'Room']])


    X_train_transformed = pd.DataFrame(enc.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]),
                                       index=X_train.index).add_prefix('enc')

    X_test_transformed = pd.DataFrame(enc.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]),
                                      index=X_test.index).add_prefix('enc')

    X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)
    X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)

    clf = fc.dtc()
    clf = clf.fit(X_train_final, y_train)

    train_score = clf.score(X_test_final, y_test)
    # End OrdinalEncoder

    # Classification Report
    report = classification_report(clf.predict(X_test_final), y_test, output_dict=True)
    print("OrdinalEncoder:" + str(round(report['macro avg']['f1-score'],2)))

    # End Classification Report

    # Begin TargetEncoder


    enc = TargetEncoder()
    enc.fit(X_train[['Zip_area', 'Room', 'Zip_loc']], y_train)

    X_train_transformed = pd.DataFrame(enc.transform(X_train[['Zip_area', 'Room', 'Zip_loc']]),
                                       index=X_train.index)

    X_test_transformed = pd.DataFrame(enc.transform(X_test[['Zip_area', 'Room', 'Zip_loc']]),
                                      index=X_test.index)

    X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)
    X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)

    clf = fc.dtc()
    clf = clf.fit(X_train_final, y_train)

    train_score = clf.score(X_test_final, y_test)
    # End TargetEncoder

    # Classification Report
    report = classification_report(clf.predict(X_test_final), y_test, output_dict=True)
    print("TargetEncoder:" + str(round(report['macro avg']['f1-score'],2)))

    # End Classification Report


