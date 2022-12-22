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


string_input = input('1 for movie by name \n'
                     '2 for movie by genre \n'
                     '3 for movie by cast \n'
                     '4 for movie by director')
menu_press = int(string_input)

if menu_press == 1:
    input1 = input('Enter your Favourite movie')
elif menu_press == 2:
    input1 = input('Enter the Genre')
elif menu_press == 3:
    input1 = input('Enter the Cast')
else:
    input1 = input('Enter name of the Director')

list_of_movies = movies_data['title'].to_list()


find_close_match = difflib.get_close_matches(input1, list_of_movies)


close_match = find_close_match[0]
index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]



similarity_score = list(enumerate(similarity[index_of_movie]))

sorted_simi = sorted(similarity_score, key=lambda x: x[1], reverse=True)



print('Movies suggested for you: ')


i = 1

for movie in sorted_simi:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if (i <= 20):
        print(i, '.', title_from_index)
        i += 1


