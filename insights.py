import csv
import os
import pandas
import ast
import operator

from json import loads

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


def get_directors():
    file_name = "credits.csv"
    dataset = pandas.read_csv(file_name)
    data = list()
    csv_name = "directing"
    crew = list()
    count = 0
    data.append("id")
    data.append("director")
    append_to_csv([data], csv_name=csv_name)
    for row in dataset.itertuples():
        data[:] = []
        count +=1
        data.append(row.id)
        movie_crew = row.crew
        movie_crew = ast.literal_eval(movie_crew)
        for crew_mem in movie_crew:
            if crew_mem["department"] == "Directing" and crew_mem["job"] == "Director":
                crew.append(crew_mem["name"].encode("utf-8"))

        data.append(crew)
        append_to_csv([data], csv_name=csv_name)
        crew = list()
        print(count)


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

def get_cast():
    file_name = "credits.csv"
    dataset = pandas.read_csv(file_name)
    data = list()
    csv_name = "cast"
    cast = list()
    count = 0
    data.append("id")
    data.append("cast_mems")
    append_to_csv([data], csv_name=csv_name)
    for row in dataset.itertuples():
        data[:] = []
        count +=1
        data.append(row.id)
        movie_cast = row.cast
        movie_cast = ast.literal_eval(movie_cast)
        for actor in movie_cast:
            cast.append(actor["name"].encode("utf-8"))

        data.append(cast)
        append_to_csv([data], csv_name=csv_name)
        cast = list()
        print(count)


def get_cast_insights(threshold):
    url = "cast.csv"
    castmems= dict()
    dataset = pandas.read_csv(url)
    for row in dataset.itertuples():
        # print(type(row.cast_mems.split('b"')[1]))
        print(row)
        test = row.cast_mems
        test = test.replace(", b'", ", '")
        test = test.replace("[b'", "['")
        test = ast.literal_eval(test)
        for person in test:
            # print(person)
            if person not in castmems:
                castmems[person] = 1
            else:
                castmems[person] = castmems[person]+ 1
    sorted_x = sorted(castmems.items(), key=operator.itemgetter(1), reverse=True)
    print(len(sorted_x))
    line_count = 0
    top_500 = list()
    for i in sorted_x:
        line_count+=1
        # print("{}: {}".format(i[0],i[1]))
        top_500.append((i[0]))
        if line_count>threshold:
            break
    return top_500


def get_director_insights(threshold):
    url = "directing.csv"
    castmems= dict()
    dataset = pandas.read_csv(url)
    for row in dataset.itertuples():
        # print(type(row.cast_mems.split('b"')[1]))
        # print(row)
        test = row.director
        test = test.replace(", b'", ", '")
        test = test.replace("[b'", "['")
        test = ast.literal_eval(test)
        for person in test:
            # print(person)
            if person not in castmems:
                castmems[person] = 1
            else:
                castmems[person] = castmems[person]+ 1
    sorted_x = sorted(castmems.items(), key=operator.itemgetter(1), reverse=True)
    print(len(sorted_x))
    print(sorted_x)
    line_count = 0
    top_500 = list()
    for i in sorted_x:
        line_count+=1
        print("{}: {}".format(i[0],i[1]))
        top_500.append((i[0]))
        if line_count>threshold:
            break
    return top_500


def get_movie_id_to_cast_dict():
    url = "cast.csv"
    castmems = dict()
    dataset = pandas.read_csv(url)
    for row in dataset.itertuples():
        # print(type(row.cast_mems.split('b"')[1]))
        test = row.cast_mems
        test = test.replace(", b'", ", '")
        test = test.replace("[b'", "['")
        test = ast.literal_eval(test)
        castmems[row.id] = test
    return castmems


def get_movie_id_to_director_dict():
    url = "directing.csv"
    castmems = dict()
    dataset = pandas.read_csv(url)
    for row in dataset.itertuples():
        # print(type(row.cast_mems.split('b"')[1]))
        test = row.director
        test = test.replace(", b'", ", '")
        test = test.replace("[b'", "['")
        test = ast.literal_eval(test)
        castmems[row.id] = test
    return castmems


def get_actor_pop_unpop_ratio(threshold):
    id_to_cast = get_movie_id_to_cast_dict()
    file_name = "movies_metadata.csv"
    dataset = pandas.read_csv(file_name)
    rogue_ids = ["1997-08-20", "2012-09-29", "2014-01-01"]
    cast_to_pop_count = dict()
    cast_to_unpop_count = dict()
    for row in dataset.itertuples():

        if row.id in rogue_ids:
            continue
        id = int(row.id)
        if id in id_to_cast:
            movie_cast = id_to_cast[id]
            for person in movie_cast:
                if row.vote_average >= 6.0:
                    if person not in cast_to_pop_count:
                        cast_to_pop_count[person] = 1
                    else:
                        cast_to_pop_count[person] += 1
                else:
                    if person not in cast_to_unpop_count:
                        cast_to_unpop_count[person] = 1
                    else:
                        cast_to_unpop_count[person] += 1

    sorted_pop = sorted(cast_to_pop_count.items(), key=operator.itemgetter(1), reverse=True)
    sorted_unpop = sorted(cast_to_unpop_count.items(), key=operator.itemgetter(1), reverse=True)
    pop_count = 0
    unpop_count = 0
    pop_cast_list = list()
    unpop_cast_list = list()
    print("POPULAR NUMBERS")
    for i in sorted_pop:
        pop_count+=1
        print("{}: {}".format(i[0], i[1]))
        pop_cast_list.append((i[0]))
        if pop_count>threshold:
            break

    print("UNPOPULAR NUMBERS")
    for i in sorted_unpop:
        unpop_count+=1
        print("{}: {}".format(i[0], i[1]))
        unpop_cast_list.append((i[0]))
        if unpop_count>threshold:
            break

    return cast_to_pop_count, cast_to_unpop_count


def get_director_pop_unpop_ratio(threshold=200):
    id_to_director = get_movie_id_to_director_dict()
    file_name = "movies_metadata.csv"
    dataset = pandas.read_csv(file_name)
    rogue_ids = ["1997-08-20", "2012-09-29", "2014-01-01"]
    director_to_pop_count = dict()
    director_to_unpop_count = dict()
    for row in dataset.itertuples():

        if row.id in rogue_ids:
            continue
        id = int(row.id)
        if id in id_to_director:
            director = id_to_director[id]
            for person in director:
                if row.vote_average >= 6.0:
                    if person not in director_to_pop_count:
                        director_to_pop_count[person] = 1
                    else:
                        director_to_pop_count[person] += 1
                else:
                    if person not in director_to_unpop_count:
                        director_to_unpop_count[person] = 1
                    else:
                        director_to_unpop_count[person] += 1

    sorted_pop = sorted(director_to_pop_count.items(), key=operator.itemgetter(1), reverse=True)
    sorted_unpop = sorted(director_to_unpop_count.items(), key=operator.itemgetter(1), reverse=True)
    pop_count = 0
    unpop_count = 0
    pop_cast_list = list()
    unpop_cast_list = list()
    print("POPULAR NUMBERS")
    for i in sorted_pop:
        pop_count+=1
        print("{}: {}".format(i[0], i[1]))
        pop_cast_list.append((i[0]))
        if pop_count>threshold:
            break

    print("UNPOPULAR NUMBERS")
    for i in sorted_unpop:
        unpop_count+=1
        print("{}: {}".format(i[0], i[1]))
        unpop_cast_list.append((i[0]))
        if unpop_count>threshold:
            break

    return director_to_pop_count, director_to_unpop_count

if __name__ == '__main__':
    get_director_pop_unpop_ratio(200)
