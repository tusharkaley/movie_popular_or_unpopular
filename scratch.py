# Load libraries
import pandas
import csv
import os
import math
from json import loads
import operator


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

def get_genres(dataset):

    genres_list = list()
    genres_dict = dict()
    for row in dataset.itertuples():
        test = row.genres
        test = test.replace("\'", "\"")
        test = loads(test)

        for a in test:
            if a["name"] not in genres_dict:
                genres_dict[a["name"]] = 1
            else:
                genres_dict[a["name"]] += 1
    sorted_x = sorted(genres_dict.items(), key=operator.itemgetter(1), reverse=True)

    for tup in sorted_x:
        if tup[1]>=700:
            genres_list.append(tup[0])
        else:
            break
    genres_list = genres_list +["other_genres"]

    return list(set(genres_list))


def get_production_houses(dataset):

    prod_houses_list = list()
    prod_houses_dict = dict()

    for row in dataset.itertuples():
        test = row.production_companies
        if type(test) == str and len(test)!= 2:
            test = test.replace("\'", "\"")
            try:
                test = loads(test)
            except:
                continue

            for a in test:
                if a["name"] not in prod_houses_dict:
                    prod_houses_dict[a["name"]] = 1
                else:
                    prod_houses_dict[a["name"]] += 1
    print(prod_houses_dict)
    sorted_x = sorted(prod_houses_dict.items(), key=operator.itemgetter(1), reverse=True)
    for tup in sorted_x:
        if tup[1]>=500:
            prod_houses_list.append(tup[0])
        else:
            break
    prod_houses_list = prod_houses_list +["other_prod_houses"]

    return prod_houses_list


def get_production_countries(dataset):

    prod_countries_list = list()
    prod_countries_dict = dict()
    ret_val = list()

    for row in dataset.itertuples():
        test = row.production_countries
        if type(test) == str and len(test)!= 2:
            test = test.replace("\'", "\"")
            try:
                test = loads(test)
                for a in test:
                    if a["name"] not in prod_countries_dict:
                        prod_countries_dict[a["name"]] = 1
                    else:
                        prod_countries_dict[a["name"]] += 1
            except:
                continue


    sorted_x = sorted(prod_countries_dict.items(), key=operator.itemgetter(1), reverse=True)
    for tup in sorted_x:
        if tup[1]>=500:
            prod_countries_list.append(tup[0])
        else:
            break
    prod_countries_list = prod_countries_list+["other_prod_countries"]
    return prod_countries_list



def transform_csv(features=["genres", "prod_companies", "prod_countries", "adult", "runtime", "original_lang", "tagline"]):

    file_name = "movies_metadata.csv"
    dataset = pandas.read_csv(file_name)
    count = 0
    genres_list = ['Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'History', 'Science Fiction', 'Mystery', 'War', 'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie']
    prod_comps = get_production_houses(dataset)
    prod_countries = get_production_countries(dataset)
    csv_results_header = [["adult", "runtime", "original_language"]+ genres_list+ prod_comps+  prod_countries+ ["class"]]
    csv_name = "movie_dataset_cleaned"
    try:
        os.remove("{}.csv".format(csv_name))
    except:
        pass
    # write_to_csv(csv_results_header, csv_name=csv_name)
    data = list()
    pop = 0
    unpop = 0
    languages = list()
    tep = list()
    # genres_dict = dict()
    temp_genres_list = list()
    temp_prod_companies_list = list()
    temp_prod_countries_list = list()
    for row in dataset.itertuples():
        data[:] = []
        print(row)

        if not math.isnan(row.vote_average) and not math.isnan(row.runtime):
            # Added adult or not
            if "adult" in features:
                if row.adult == "False":
                    data.append(0)
                else:
                    data.append(1)

            # Added runtime
            if "runtime" in features:
                data.append(row.runtime)
            if "original_lang" in features:
                if row.original_language not in languages:
                    languages.append(row.original_language)
                    data.append(languages.index(row.original_language))
                else:
                    data.append(languages.index(row.original_language))

            # Populating the genres
            if "genres" in features:
                test = row.genres
                test = test.replace("\'", "\"")
                test = loads(test)
                for a in test:
                    temp_genres_list.append(a["name"])
                temp_genres_list = list(set(temp_genres_list))
                for gen in genres_list:
                    if gen in temp_genres_list:
                        data.append(1)
                    else:
                        data.append(0)
                temp_genres_list = list()

            # Populating the production companies
            if "prod_companies" in features:
                test = row.production_companies
                process_prod_comp = True
                pc_added = False
                if type(test) == str and len(test) != 2:
                    test = test.replace("\'", "\"")
                    try:
                        test = loads(test)
                    except:
                        pass
                        process_prod_comp = False
                    if process_prod_comp:
                        for a in test:
                            temp_prod_companies_list.append(a["name"].encode("utf-8"))
                        for pc in prod_comps[:-1]:
                            if pc in temp_prod_companies_list:
                                data.append(1)
                                pc_added = True
                            else:
                                data.append(0)
                        # Accounting for the other countries feature
                        if not pc_added:
                            data.append(1)
                        else:
                            data.append(0)
                    else:
                        for pc in prod_comps:
                            if pc == "other_prod_houses":
                                data.append(1)
                            else:
                                data.append(0)
                else:
                    for pc in prod_comps:
                        if pc == "other_prod_houses":
                            data.append(1)
                        else:
                            data.append(0)

                temp_prod_companies_list = list()

            # Populating the production countries
            if "prod_countries" in features:
                test = row.production_countries
                test = test.replace("\'", "\"")
                country_added = False
                if type(test) == str and len(test) != 2:
                    test = test.replace("\'", "\"")
                    try:
                        test = loads(test)
                        for a in test:
                            temp_prod_countries_list.append(a["name"])
                        for pc in prod_countries[:-1]:
                            if pc in temp_prod_countries_list:
                                data.append(1)
                                country_added = True
                            else:
                                data.append(0)
                        # Accounting for the other countries feature
                        if not country_added:
                            data.append(1)
                        else:
                            data.append(0)
                    except:
                        for pc in prod_countries:
                            if pc == "other_prod_countries":
                                data.append(1)
                            else:
                                data.append(0)
                        pass
                else:
                    for pc in prod_countries:
                        if pc == "other_prod_countries":
                            data.append(1)
                        else:
                            data.append(0)

            if "tagline" in features:
                if row.tagline == str and len(row.tagline) > 0:
                    data.append(1)
                else:
                    data.append(0)
            if row.vote_average >= 6.0:
                pop = pop + 1
                data.append(0)
            else:
                unpop = unpop + 1
                data.append(1)

            count = count +1
            tep.append(len(data))
            append_to_csv([data], csv_name=csv_name)

    print(count)
    tep = list(set(tep))
    print("lenght of temp")
    print(tep)
    print("count: {}, popular: {}, unpopular: {}".format(count, pop, unpop))


def classify_version_1():
    file_name = "movies_metadata.csv"
    dataset = pandas.read_csv(file_name)
    prod_comps = get_production_houses(dataset)
    prod_countries = get_production_countries(dataset)
    url = "movie_dataset_cleaned.csv"
    names= ["runtime", "original_language", 'Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'History', 'Science Fiction', 'Mystery', 'War', 'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie']+prod_comps+prod_countries+ ["class"]
    # names = ["adult", "runtime", "original_language", "class"]
    dataset = pandas.read_csv(url, names=names, usecols=names)
    print(dataset.shape)
    # print(dataset.groupby('original_language').size())
    array = dataset.values
    print(array.shape)
    scoring = 'accuracy'

    X = array[:, 0:array.shape[1]-1]
    Y = array[:, array.shape[1]-1]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC(gamma='auto')))
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
    #
    # print(clf.predict([[0, 150, 1]]))
    # print(clf.predict([[0, 90, 1]]))
    # print(clf.predict([[0, 120, 1]]))
    # print(clf.predict([[0, 30, 1]]))
    # print(clf.predict([[0, 60, 1]]))
    # print(clf.predict([[0, 190, 1]]))


if __name__ == '__main__':
    # features = ["genres", "prod_companies", "prod_countries", "runtime", "original_lang", "popularity"]
    # transform_csv(features)
    # classify_version_1()
    file_name = "credits.csv"
    dataset = pandas.read_csv(file_name)
    print(dataset.head(10))
    # print(get_production_houses(dataset))
    # print(get_production_countries(dataset))

    # print(os.getcwd())