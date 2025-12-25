from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import feature as fe
from sklearn.svm import LinearSVC

def Logistic_Regression(data):

    features = fe.get_structual_features(data)

    X_struct = data[features]  # dense Data Frame
    tfidf, X_tfidf, y = fe.tfid_data(data)  # sparse matrix
    scaler = StandardScaler()  # scaler of data
    X_struct_scaled = scaler.fit_transform(X_struct)  # N (0,ham)

    X_combined = hstack([X_tfidf, X_struct_scaled])  # combine TFIDF and Structure features
    X_train, X_test, y_train, y_test = train_test_split(  # divide data and label
        X_combined,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)  # train model

    y_pred = model.predict(X_test)  # model predict

    return y_test, y_pred

def Linear_SVC(data):

    features = fe.get_structual_features(data)

    X_struct = data[features]  # dense Data Frame
    tfidf, X_tfidf, y = fe.tfid_data(data)  # sparse matrix
    scaler = StandardScaler()  # scaler of data
    X_struct_scaled = scaler.fit_transform(X_struct)  # N (0,ham)

    X_combined = hstack([X_tfidf, X_struct_scaled])  # combine TFIDF and Structure features
    X_train, X_test, y_train, y_test = train_test_split(  # divide data and label
        X_combined,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LinearSVC(class_weight='balanced') #model LinearSVC
    model.fit(X_train, y_train) #model training

    y_pred = model.predict(X_test) #model predict

    return y_test, y_pred

def Logistic_Regression_proba(data):

    features = fe.get_structual_features(data)

    X_struct = data[features]  # dense Data Frame
    tfidf, X_tfidf, y = fe.tfid_data(data)  # sparse matrix
    scaler = StandardScaler()  # scaler of data
    X_struct_scaled = scaler.fit_transform(X_struct)  # N (0,ham)

    X_combined = hstack([X_tfidf, X_struct_scaled])  # combine TFIDF and Structure features
    X_train, X_test, y_train, y_test = train_test_split(  # divide data and label
        X_combined,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)  # train model

    y_proba = model.predict_proba(X_test)[:, 1]

    return y_test, y_proba

def Logistic_Regression_test(data1, data2):

    features1 = fe.get_structual_features(data1)
    features2 = fe.get_structual_features(data2)
    features = list(set(features1) & set(features2))

    scaler = StandardScaler()# scaler of data

    X_struct1 = data1[features]  # dense Data Frame
    tfidf, X_tfidf1, y1, X_tfidf2, y2 = fe.tfid_two_data(data1, data2)  # sparse matrix
    X_struct_scaled1 = scaler.fit_transform(X_struct1)  # N (0,ham)
    X_combined1 = hstack([X_tfidf1, X_struct_scaled1])  # combine TFIDF and Structure features

    X_struct2 = data2[features]  # dense Data Frame
    X_struct_scaled2 = scaler.transform(X_struct2)  # N (0,ham)
    X_combined2 = hstack([X_tfidf2, X_struct_scaled2])  # combine TFIDF and Structure features

    X_train = X_tfidf1  # X_combined1
    X_test = X_tfidf2  # X_combined2
    y_train = y1
    y_test = y2

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)  # train model

    y_pred = model.predict(X_test)  # model predict

    return y_test, y_pred


def Linear_SVC_test(data1, data2):

    features1 = fe.get_structual_features(data1)
    features2 = fe.get_structual_features(data2)
    features = list(set(features1) & set(features2))

    scaler = StandardScaler()  # scaler of data

    X_struct1 = data1[features]  # dense Data Frame
    tfidf, X_tfidf1, y1, X_tfidf2, y2 = fe.tfid_two_data(data1, data2)
    X_struct_scaled1 = scaler.fit_transform(X_struct1)  # N (0,ham)
    X_combined1 = hstack([X_tfidf1, X_struct_scaled1])  # combine TFIDF and Structure features

    X_struct2 = data2[features]  # dense Data Frame  # sparse matrix
    X_struct_scaled2 = scaler.transform(X_struct2)  # N (0,ham)
    X_combined2 = hstack([X_tfidf2, X_struct_scaled2])  # combine TFIDF and Structure features

    X_train = X_tfidf1 #X_combined1
    X_test = X_tfidf2 #X_combined2
    y_train = y1
    y_test = y2

    model = LinearSVC(class_weight='balanced')  # model LinearSVC
    model.fit(X_train, y_train)  # model training

    y_pred = model.predict(X_test)  # model predict

    return y_test, y_pred