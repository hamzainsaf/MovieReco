import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_data = pd.read_csv('movies.csv')
# print(movies_data.head())

select_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in select_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + \
                    movies_data['cast'] + ' ' + movies_data['director']
# print(combined_features)

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
# print(feature_vectors)


similarity = cosine_similarity(feature_vectors)

# print(similarity.shape)
list_of_movies = movies_data['title'].to_list()
list_of_movies_g = movies_data['genres'].to_list()
list_of_movies_c = movies_data['cast'].to_list()
list_of_movies_d = movies_data['director'].to_list()


def check(column_name, input, list1):
    find_close_match = difflib.get_close_matches(input, list1)

    close_match = find_close_match[0]

    index_of_movie = movies_data[getattr(movies_data, column_name) == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_movie]))
    sorted_simi = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    print('Movies suggested for you: ')

    i = 1

    for movie in sorted_simi:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        if i <= 10:
            print(i, '.', title_from_index)
            i += 1


menu_press = 10
while menu_press != 0:
    string_input = input('1 for movie by name \n'
                         '2 for movie by genre \n'
                         '3 for movie by cast \n'
                         '4 for movie by director\n'
                         '0 for Quit\n'
                         )

    menu_press = int(string_input)

    if menu_press == 1:
        c_name = 'title'
        input1 = input('Enter your Favourite movie \n')
        check(c_name, input1, list_of_movies)


    elif menu_press == 2:
        c_name = 'genres'
        input1 = input('Enter the Genre \n')
        check(c_name, input1, list_of_movies_g)


    elif menu_press == 3:
        c_name = 'cast'
        input1 = input('Enter the Cast\n')
        check(c_name, input1, list_of_movies_c)

    elif menu_press == 4:
        c_name = 'director'
        input1 = input('Enter name of the Director\n')
        check(c_name, input1, list_of_movies_d)
