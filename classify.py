# Load libraries
import pandas
import csv
import os
import math
from json import loads
import operator
import ast
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

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
    print("Getting the list of popular production companies")
    prod_comps = ins.get_production_houses(dataset)
    print("Getting the list of popular production countries")
    prod_countries = ins.get_production_countries(dataset)
    print("Getting the data on popular and unpopular movie numbers on cast")
    pop_cast_list, unpop_cast_list = ins.get_actor_pop_unpop_ratio(cast_threshold)
    print("Getting the data on popular and unpopular movie numbers on directors")
    pop_dir_list, unpop_dir_list = ins.get_director_pop_unpop_ratio(cast_threshold)
    print("Getting the data on popular and unpopular movie numbers on writers")
    pop_writers_list, unpop_writers_list = ins.get_writers_pop_unpop_ratio(cast_threshold)
    # pop_prod_comps , unpop_prod_comps = ins.get_prod_comps_pop_unpop_ratio()

    id_to_cast = ins.get_movie_id_to_cast_dict()
    id_to_director = ins.get_movie_id_to_director_dict()
    id_to_writer = ins.get_movie_id_to_writer_dict()
    # csv_results_header = [["adult", "runtime", "original_language"] + genres_list + prod_comps + prod_countries + ["class"]]
    csv_name = "movie_dataset_cleaned"
    try:
        os.remove("{}.csv".format(csv_name))
    except:
        pass
    # write_to_csv(csv_results_header, csv_name=csv_name)
    data = list()
    pop = 0
    unpop = 0
    cast_processed = 0
    pc_processed = 0
    languages = list()
    tep = list()
    # genres_dict = dict()
    temp_genres_list = list()
    temp_prod_companies_list = list()
    temp_prod_countries_list = list()
    tmp_cnt = 0
    pro_count_pop = list()
    pro_count_unpop = list()
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
            # Added Original Language
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
                        for pc in prod_comps:
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
                            data.append(0)
                else:
                    for pc in prod_comps:
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

            if "cast_score" in features:
                if id in id_to_cast:
                    cast_processed += 1
                    movie_dir = id_to_cast[id]
                    cast_score = 0
                    for person in movie_dir:
                        if person in pop_cast_list:
                            cast_score = cast_score + pop_cast_list[person]
                        if person in unpop_cast_list:
                            cast_score = cast_score - unpop_cast_list[person]
                    if cast_score < 0:
                        data.append(0)
                    else:
                        data.append(1)
                else:
                    tmp_cnt += 1
                    data.append(0)

                # print("top_count: {}".format(top_count))

            if "directors_score" in features:
                if id in id_to_director:
                    movie_dir = id_to_director[id]
                    direct_score = 0
                    for person in movie_dir:
                        if person in pop_dir_list:
                            direct_score = direct_score + pop_dir_list[person]
                        if person in unpop_dir_list:
                            direct_score = direct_score - unpop_dir_list[person]
                    if direct_score < 0:
                        data.append(0)
                    else:
                        data.append(1)
                else:
                    tmp_cnt += 1
                    data.append(0)

            if "writers_score" in features:
                if id in id_to_writer:
                    movie_writer = id_to_writer[id]
                    writer_score = 0
                    for person in movie_writer:
                        if person in pop_writers_list:
                            writer_score = writer_score + pop_writers_list[person]
                        if person in unpop_writers_list:
                            writer_score = writer_score - unpop_writers_list[person]
                    if writer_score < 0:
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
                    data.append(1)
            if row.vote_average >= 6.0:
                pop = pop + 1
                data.append(1)
            else:
                unpop = unpop + 1
                data.append(0)

            count = count +1
            if count % 5000 ==0:
                print("{} movies processed".format(count))
            # print(count)
            tep.append(len(data))
            ins.append_to_csv([data], csv_name=csv_name)

    tep = list(set(tep))

    print("Total number of movies: {},\n Popular Movies: {}\n, Unpopular Movies: {}".format(count, pop, unpop))


def train_and_cross_validate():
    file_name = "movies_metadata.csv"
    dataset = pandas.read_csv(file_name)
    prod_comps = ins.get_production_houses(dataset)
    prod_countries = ins.get_production_countries(dataset)
    genres_list = ins.get_genres(dataset)
    url = "movie_dataset_cleaned.csv"
    names = ["runtime", "original_lang"] + genres_list + prod_comps + prod_countries + ["cast_score", "directors_score", "writers_score",  "belongs_to_collection"] + ["class"]
    dataset = pandas.read_csv(url, names=names, usecols=names)
    array = dataset.values
    scoring = 'accuracy'

    X = array[:, 0:array.shape[1]-1]
    Y = array[:, array.shape[1]-1]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    models = list()

    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='auto')))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC(gamma='auto')))
    results = list()
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results.mean()*100)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    print("SCORES\n")
    knn = LogisticRegression(solver='liblinear', multi_class='auto')
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))



if __name__ == '__main__':
    features = ["genres", "prod_companies", "prod_countries", "runtime", "original_lang", "popularity", "cast_score", "directors_score", "writers_score", "belongs_to_collection"]
    runtime = 100
    transform_csv(features, runtime, 100)
    train_and_cross_validate()
