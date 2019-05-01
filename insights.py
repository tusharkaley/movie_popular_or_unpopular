import csv
import os
import pandas
import ast
import operator
import matplotlib.pyplot as plt
import numpy as np

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


def get_writers():
    file_name = "credits.csv"
    dataset = pandas.read_csv(file_name)
    data = list()
    csv_name = "writing"
    crew = list()
    count = 0
    data.append("id")
    data.append("writer")
    append_to_csv([data], csv_name=csv_name)
    for row in dataset.itertuples():
        data[:] = []
        count +=1
        data.append(row.id)
        movie_crew = row.crew
        movie_crew = ast.literal_eval(movie_crew)
        for crew_mem in movie_crew:
            if crew_mem["department"] == "Writing" and crew_mem["job"] == "Screenplay":
                crew.append(crew_mem["name"].encode("utf-8"))

        data.append(crew)
        append_to_csv([data], csv_name=csv_name)
        crew = list()


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
    sorted_x = sorted(prod_houses_dict.items(), key=operator.itemgetter(1), reverse=True)
    for tup in sorted_x:
        if tup[1]>=500:
            prod_houses_list.append(tup[0])
        else:
            break
    prod_houses_list = prod_houses_list

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
    line_count = 0
    top_500 = list()
    for i in sorted_x:
        line_count+=1
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
    directors = dict()
    dataset = pandas.read_csv(url)
    for row in dataset.itertuples():
        # print(type(row.cast_mems.split('b"')[1]))
        test = row.director
        test = test.replace(", b'", ", '")
        test = test.replace("[b'", "['")
        test = ast.literal_eval(test)
        directors[row.id] = test
    return directors


def get_movie_id_to_writer_dict():
    url = "writing.csv"
    writers = dict()
    dataset = pandas.read_csv(url)
    for row in dataset.itertuples():
        # print(type(row.cast_mems.split('b"')[1]))
        test = row.writer
        test = test.replace(", b'", ", '")
        test = test.replace("[b'", "['")
        test = ast.literal_eval(test)
        writers[row.id] = test
    return writers


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
    for i in sorted_pop:
        pop_count+=1
        pop_cast_list.append((i[0]))
        if pop_count>threshold:
            break
    for i in sorted_unpop:
        unpop_count+=1
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
    for i in sorted_pop:
        pop_count+=1
        pop_cast_list.append((i[0]))
        if pop_count > threshold:
            break
    for i in sorted_unpop:
        unpop_count+=1
        unpop_cast_list.append((i[0]))
        if unpop_count>threshold:
            break

    return director_to_pop_count, director_to_unpop_count


def get_writers_pop_unpop_ratio(threshold):

    id_to_writer = get_movie_id_to_writer_dict()
    file_name = "movies_metadata.csv"
    dataset = pandas.read_csv(file_name)
    rogue_ids = ["1997-08-20", "2012-09-29", "2014-01-01"]
    writer_to_pop_count = dict()
    writer_to_unpop_count = dict()
    for row in dataset.itertuples():

        if row.id in rogue_ids:
            continue
        id = int(row.id)
        if id in id_to_writer:
            movie_cast = id_to_writer[id]
            for person in movie_cast:
                if row.vote_average >= 6.0:
                    if person not in writer_to_pop_count:
                        writer_to_pop_count[person] = 1
                    else:
                        writer_to_pop_count[person] += 1
                else:
                    if person not in writer_to_unpop_count:
                        writer_to_unpop_count[person] = 1
                    else:
                        writer_to_unpop_count[person] += 1

    sorted_pop = sorted(writer_to_pop_count.items(), key=operator.itemgetter(1), reverse=True)
    sorted_unpop = sorted(writer_to_unpop_count.items(), key=operator.itemgetter(1), reverse=True)
    pop_count = 0
    unpop_count = 0
    pop_cast_list = list()
    unpop_cast_list = list()
    for i in sorted_pop:
        pop_count+=1
        pop_cast_list.append((i[0]))
        if pop_count>threshold:
            break

    for i in sorted_unpop:
        unpop_count+=1
        unpop_cast_list.append((i[0]))
        if unpop_count>threshold:
            break

    return writer_to_pop_count, writer_to_unpop_count


def get_prod_comps_pop_unpop_ratio():

    file_name = "movies_metadata.csv"
    dataset = pandas.read_csv(file_name)
    prod_comps_to_pop_count = dict()
    prod_comps_to_unpop_count = dict()
    count = 0
    for row in dataset.itertuples():
        test = row.production_companies
        process_prod_comp = True
        # pc_added = False
        temp_prod_companies_list = list()
        if type(test) == str and len(test) != 2:
            test = test.replace("\'", "\"")
            try:
                test = loads(test)
            except:
                pass
                process_prod_comp = False
            if process_prod_comp:
                count +=1
                for a in test:
                    temp_prod_companies_list.append(a["name"])
                for pc in temp_prod_companies_list:
                    if row.vote_average >= 6.0:
                        if pc not in prod_comps_to_pop_count:
                            prod_comps_to_pop_count[pc] = 1
                        else:
                            prod_comps_to_pop_count[pc] += 1
                    else:
                        if pc not in prod_comps_to_unpop_count:
                            prod_comps_to_unpop_count[pc] = 1
                        else:
                            prod_comps_to_unpop_count[pc] += 1
    return prod_comps_to_pop_count, prod_comps_to_unpop_count


def movie_runtime_plot():
    file_name = "movies_metadata.csv"
    dataset = pandas.read_csv(file_name)
    x = list()
    y = list()
    for row in dataset.itertuples():
        y.append(row.runtime)
        x.append(row.vote_average)
    plt.ylim(0, 200)
    plt.scatter(x, y)
    plt.show()


def prod_comps_plot(pop_list):

    pop_list.sort()
    seed = 100
    offset = 100
    count = 0
    x = list()
    for p in pop_list:
        if p < seed:
            count+=1
        else:
            seed = seed + offset
            x.append(count)
            count = 0
    print(x)


def prod_comps():
    objects = ('<100', '100-200', '200-300', '300-400', '400-500', '500-600', '700-800', '800-900', '900-1000', '1000-1100', '1100-1200', '1200-1300', '1300-1400','1400-1500', '1500-1600')
    y_pos = np.arange(len(objects))
    performance = [12417, 1341, 879, 158, 523, 109, 71, 26, 868, 99, 1126, 35, 580, 132, 32]

    plt.figure(figsize=(15, 5))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Distribution of popular movies')
    plt.title('Number of production houses')

    plt.show()


def plot_top_companies():

    file_name = "movies_metadata.csv"
    dataset = pandas.read_csv(file_name)
    prod_houses_list = list()
    prod_houses_nums = list()
    prod_houses_dict = dict()

    for row in dataset.itertuples():
        test = row.production_companies
        if type(test) == str and len(test) != 2:
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
    sorted_x = sorted(prod_houses_dict.items(), key=operator.itemgetter(1), reverse=True)
    for tup in sorted_x:
        if tup[1] >= 500:
            prod_houses_list.append(tup[0])
            prod_houses_nums.append(tup[1])
        else:
            break
    objects = prod_houses_list
    y_pos = np.arange(len(objects))
    performance = prod_houses_nums

    plt.figure(figsize=(15, 10))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Number of movies')
    plt.title('Production houses to number of movies')

    plt.show()


def plot_top_countries():

    file_name = "movies_metadata.csv"
    dataset = pandas.read_csv(file_name)
    prod_houses_list = list()
    prod_houses_nums = list()
    prod_houses_dict = dict()

    for row in dataset.itertuples():
        test = row.production_countries
        if type(test) == str and len(test) != 2:
            test = test.replace("\'", "\"")
            try:
                test = loads(test)
                for a in test:
                    if a["name"] not in prod_houses_dict:
                        prod_houses_dict[a["name"]] = 1
                    else:
                        prod_houses_dict[a["name"]] += 1
            except:
                continue

    sorted_x = sorted(prod_houses_dict.items(), key=operator.itemgetter(1), reverse=True)
    for tup in sorted_x:
        if tup[1] >= 500:
            prod_houses_list.append(tup[0])
            prod_houses_nums.append(tup[1])
        else:
            break
    objects = prod_houses_list
    y_pos = np.arange(len(objects))
    performance = prod_houses_nums

    plt.figure(figsize=(15, 10))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Number of movies')
    plt.title('Production country to number of movies')

    plt.show()


def plot_cast_comparison(means_popular, means_unpopular, cast):
    n_groups = 4

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.7

    rects1 = plt.bar(index, means_popular, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Popular')

    rects2 = plt.bar(index + bar_width, means_unpopular, bar_width,
                     alpha=opacity,
                     color='r',
                     label='Unpopular')

    plt.xlabel('Writer')
    plt.ylabel('Number of movies')
    plt.title('Number of popular and unpopular movies')
    plt.xticks(index + bar_width, cast)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plt_lines():
    X = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
    Y = [57.92, 58.53, 63.01, 85.51, 89.67, 74.80, 79.61, 86.44, 86.35]
    plt.plot(X, Y)
    plt.show()


def plot_pie():

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Popular', 'Unpopular',
    sizes = [53.41, 46.59]
    explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()


if __name__ == '__main__':
    file_name = "movies_metadata.csv"
    dataset = pandas.read_csv(file_name)
    get_production_countries(dataset)
        # get_director_pop_unpop_ratio(200)
        # movie_runtime_plot()
        # prod_comps()
        # plot_top_countries()
        # get_actor_pop_unpop_ratio(100)
        # get_director_pop_unpop_ratio(200)
        # get_writers_pop_unpop_ratio(200)
        # means_popular = (17, 18, 8, 13)
        # means_unpopular = (11, 11, 6, 6)
        # cast = ('Julius J. Epstein', 'Dudley Nichols', 'Everett Freeman', 'Sylvester Stallone')
        # plot_cast_comparison(means_popular, means_unpopular, cast)
        # plt_lines()
        # plt_lines()