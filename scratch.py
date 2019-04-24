# Load libraries
import pandas
import csv
import os
import math

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Load dataset
# url = "movies_metadata.csv"
# names = ['runtime', 'genres']
# dataset = pandas.read_csv(url, usecols=names)
# print("Tushar")
# print(dataset.shape)
# # print(dataset.head(20))
# count = 0
# for i in dataset:
#     print(i)
#     count = count + 1
#     if count == 20:
#         break

# print(dataset.describe())
# print(dataset.groupby('class').size())
#
# # dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# # plt.show()
# # scatter plot matrix
# # scatter_matrix(dataset)
# # plt.show()
#
# # Split-out validation dataset
# array = dataset.values
# X = array[:, 0:4]
# Y = array[:, 4]
# validation_size = 0.20
# seed = 7
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#
# # Test options and evaluation metric
# seed = 7
# scoring = 'accuracy'
#
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# # evaluate each model in turn
# results = []
# names = []
# for name, model in models:
#     kfold = model_selection.KFold(n_splits=10, random_state=seed)
#     cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)
#
# # Compare Algorithms
# # fig = plt.figure()
# # fig.suptitle('Algorithm Comparison')
# # ax = fig.add_subplot(111)
# # plt.boxplot(results)
# # ax.set_xticklabels(names)
# # plt.show()
#
# knn = KNeighborsClassifier()
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))


def write_to_csv(data, csv_name="output", csv_dir="/Users/tusharkale/Documents/UF/Workspace/python_workspace/movie_popular_or_unpopular"):
    """
        Function to cleanup the labels added by the 'most_popular_numbers' function
        Args:
            data: Data to write to csv
            csv_name: Name of the csv to write
            csv_dir: where to save the csv
    """

    try:
        print(data)
        filename = '%s.csv' % csv_name
        file_path = os.path.join(csv_dir, filename)
        with open(file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerows(data)

    except Exception as exc:
        print("write_to_csv issue")
        raise exc


def append_to_csv(data, csv_name="output", csv_dir="/Users/tusharkale/Documents/UF/Workspace/python_workspace/movie_popular_or_unpopular"):
    """
        Function to cleanup the labels added by the 'most_popular_numbers' function
        Args:
            data: Data to write to csv
            csv_name: Name of the csv to write
            csv_dir: where to save the csv
    """

    try:

        filename = '%s.csv' % csv_name
        file_path = os.path.join(csv_dir, filename)
        with open(file_path, "a") as f:
            writer = csv.writer(f)
            writer.writerows(data)

    except Exception as exc:
        print("append_to_csv issue")
        raise exc


def transform_csv():

    file_name = "movies_metadata.csv"
    dataset = pandas.read_csv(file_name)
    count = 0
    csv_results_header = [[0, 121, 2, "popular"]]
    csv_name = "movie_dataset_cleaned"
    write_to_csv(csv_results_header, csv_name=csv_name)
    data = list()
    pop = 0
    unpop = 0
    languages = list()

    for row in dataset.itertuples():
        data[:] = []
        print(row)
        if not math.isnan(row.vote_average) and not math.isnan(row.runtime):
            if row.adult == "False":
                data.append(0)
            else:
                data.append(1)
            data.append(row.runtime)
            if row.original_language not in languages:
                languages.append(row.original_language)
                data.append(languages.index(row.original_language))
            else:
                data.append(languages.index(row.original_language))

            # data.append(row.genres)
            # data.append(row.vote_average)
            if row.vote_average >= 6.0:
                pop = pop + 1
                data.append("popular")
            else:
                unpop = unpop + 1
                data.append("unpopular")

            count = count +1
            append_to_csv([data], csv_name=csv_name)

    print(count)
    print("count: {}, popular: {}, unpopular: {}".format(count, pop, unpop))


def classify_version_1():
    url = "movie_dataset_cleaned.csv"
    names= ["adult", "runtime", "class"]
    dataset = pandas.read_csv(url, names=names, usecols=names)
    print(dataset.shape)
    # print(dataset.groupby('original_language').size())
    array = dataset.values
    print(array)
    scoring = 'accuracy'

    X = array[:, 0:2]
    Y = array[:, 2]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    models = []
    # models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    # # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # clf = SVC(gamma='auto')
    # clf.fit(X_train, Y_train)

    # print(clf.predict([[0, 121]]))

if __name__ == '__main__':
    # transform_csv()
    classify_version_1()
