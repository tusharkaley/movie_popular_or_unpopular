# Load libraries
import pandas
import csv
import os
import math
from json import loads
import operator
import ast

import movie_popular_or_unpopular.insights as ins

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


def transform_csv(features=["genres", "prod_companies", "prod_countries", "adult", "runtime", "original_lang", "tagline"], runtime_thres = 100, cast_threshold= 500):

    file_name = "movies_metadata.csv"
    dataset = pandas.read_csv(file_name)
    count = 0
    collection = 0
    genres_list = ins.get_genres(dataset)
    prod_comps = ins.get_production_houses(dataset)
    prod_countries = ins.get_production_countries(dataset)
    pop_cast_list, unpop_cast_list = ins.get_actor_pop_unpop_ratio(cast_threshold)
    pop_dir_list, unpop_dir_list = ins.get_director_pop_unpop_ratio(cast_threshold)
    top_dirs = ins.get_director_insights(cast_threshold)
    id_to_cast = ins.get_movie_id_to_cast_dict()
    id_to_director = ins.get_movie_id_to_director_dict()
    csv_results_header = [["adult", "runtime", "original_language"] + genres_list + prod_comps + prod_countries + ["class"]]
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
    tmp_cnt = 0
    rogue_ids = ["1997-08-20" , "2012-09-29", "2014-01-01"]
    for row in dataset.itertuples():
        data[:] = []

        if row.id in rogue_ids:
            continue
        id = int(row.id)

        if not math.isnan(row.vote_average) and not math.isnan(row.runtime):
            # Added adult or not
            if "adult" in features:
                if row.adult == "False":
                    data.append(0)
                else:
                    data.append(1)

            # Added runtime
            if "runtime" in features:
                if int(row.runtime) >= runtime_thres:
                    data.append(1)
                else:
                    data.append(0)

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
            top_count = 0

            if "top_500" in features:
                if id in id_to_cast:
                    movie_dir = id_to_cast[id]
                    cast_score = 0
                    for person in movie_dir:
                        if person in pop_cast_list:
                            cast_score = cast_score + pop_cast_list[person]
                        if person in unpop_cast_list:
                            cast_score = cast_score - unpop_cast_list[person]
                    if cast_score <= 0:
                        data.append(0)
                    else:
                        data.append(1)
                else:
                    tmp_cnt += 1
                    data.append(0)

            if "top_count" in features:
                data.append(top_count)
                # print("top_count: {}".format(top_count))

            if "top_directors" in features:
                if id in id_to_director:
                    movie_dir = id_to_director[id]
                    direct_score = 0
                    for person in movie_dir:
                        if person in pop_dir_list:
                            direct_score = direct_score + pop_dir_list[person]
                        if person in unpop_dir_list:
                            direct_score = direct_score - unpop_dir_list[person]
                    if direct_score <= 0:
                        data.append(0)
                    else:
                        data.append(1)
                else:
                    tmp_cnt += 1
                    data.append(0)

            if "belongs_to_collection" in features:
                if type(row.belongs_to_collection) != str:
                    data.append(0)
                else:
                    collection += 1
                    print("belongs to collection")
                    data.append(1)
            if row.vote_average >= 6.0:
                pop = pop + 1
                data.append(0)
            else:
                unpop = unpop + 1
                data.append(1)

            count = count +1
            # print(count)
            tep.append(len(data))
            ins.append_to_csv([data], csv_name=csv_name)

    print(count)
    print("Movies in collection : {}".format(collection))

    tep = list(set(tep))
    print("length of temp")
    print(tep)
    print("count: {}, popular: {}, unpopular: {}".format(count, pop, unpop))


def train_and_cross_validate():
    file_name = "movies_metadata.csv"
    dataset = pandas.read_csv(file_name)
    prod_comps = ins.get_production_houses(dataset)
    prod_countries = ins.get_production_countries(dataset)
    genres_list = ins.get_genres(dataset)
    url = "movie_dataset_cleaned.csv"
    names = ["runtime", "original_language"] + genres_list + prod_comps + prod_countries +["top_500","top_count", "top_directors", "belongs_to_collection"]+["class"]
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

    models = list()

    # models.append(('ovr-liblinear', LogisticRegression(solver='liblinear', multi_class='ovr')))
    # models.append(('ovr-newton - cg', LogisticRegression(solver='newton-cg', multi_class='ovr')))
    # models.append(('ovr-lbfgs', LogisticRegression(solver='lbfgs', multi_class='ovr')))
    # models.append(('ovr-sag', LogisticRegression(solver='sag', multi_class='ovr', max_iter=1000)))
    # models.append(('ovr-saga', LogisticRegression(solver='saga', multi_class='ovr', max_iter=500)))
    #
    # models.append(('multinomial-newton-cg', LogisticRegression(solver='newton-cg', multi_class='multinomial')))
    # models.append(('multinomial-lbfgs', LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=500)))
    # models.append(('multinomial-sag', LogisticRegression(solver='sag', multi_class='multinomial')))
    # models.append(('multinomial-saga', LogisticRegression(solver='saga', multi_class='multinomial')))

    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='auto')))
    # models.append(('LR', LogisticRegression(solver='newton-cg', multi_class='auto')))
    # models.append(('LR', LogisticRegression(solver='lbfgs', multi_class='auto')))
    # models.append(('LR', LogisticRegression(solver='sag', multi_class='auto')))
    # models.append(('LR', LogisticRegression(solver='saga', multi_class='auto')))

    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
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
    print("SCORES")
    knn = LogisticRegression(solver='liblinear', multi_class='ovr')
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


if __name__ == '__main__':
    features = ["genres", "prod_companies", "prod_countries", "runtime", "original_lang", "popularity", "top_500", "top_count", "top_directors", "belongs_to_collection"]
    runtime = 100
    transform_csv(features, runtime, 100)
    train_and_cross_validate()
