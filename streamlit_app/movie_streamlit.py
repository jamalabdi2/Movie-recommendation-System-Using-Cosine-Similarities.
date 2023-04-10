import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import difflib
from PIL import Image
import requests
from io import BytesIO

# title
st.title('Movie Recommender')
st.header("Please Enter Your favorite movie and I will suggest you related Movies: ")

def load_data():
    movie_dataframe = pd.read_json('/Users/jamal/Golang/cleaned_movie_dataframe.json')
    return movie_dataframe

movie_dataframe = load_data()

def join_strings(df, column_name):
    df.loc[:, column_name] = df[column_name].apply(lambda x: ' '.join(x))
    return df

join_strings(movie_dataframe,'Genre')
join_strings(movie_dataframe,'Stars')

def combine_features(movies):
    # Initialize an empty string to store the combined features
    combined_features = ""

    # Iterate through each feature and concatenate its values with a space separator
    for column in movies.columns:
        # Fill any missing values with an empty string and convert to string type
        combined_features += movies[column].fillna('').astype(str) + ' '

    # Strip any leading/trailing whitespace and return the combined string as a pandas Series
    return combined_features.str.strip()

selected_columns = ['Genre','Director','Stars','Rating','Year','Description']
selected_features = movie_dataframe[selected_columns]
combined_features = combine_features(selected_features)

def get_feature_vector(data):
    """
    Convert a list of text data into a sparse matrix of TF-IDF vectors.

    Args:
    data (list): A list of strings to be vectorized.

    Returns:
    scipy.sparse.csr_matrix: A sparse matrix of shape (n_samples, n_features) representing the TF-IDF vectors of the input data.

    """

    # Define a TfidfVectorizer object to convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the input data and transform it into a sparse matrix
    feature_vector = vectorizer.fit_transform(data)
    return feature_vector

feature_vector = get_feature_vector(combined_features)

def get_cosine_similarity_matrix(feature_vector):
    similarity_matrix = cosine_similarity(feature_vector)
    return similarity_matrix

similarity = get_cosine_similarity_matrix(feature_vector)
title_2 = [ re.sub(r'\W+','',title.lower()) for title in movie_dataframe['Title']]
movie_dataframe['Title2'] = title_2

def get_movie_posters(similar_movies,movie_dataframe):
    movie_posters = []
    movie_titles = []
    for movie in similar_movies[:12]:
        index = movie[0]
        movie_title = movie_dataframe.iloc[index].Title
        movie_poster = movie_dataframe.iloc[index]['Movie Poster']
        movie_posters.append(movie_poster)
        movie_titles.append(movie_title)
      
    return movie_titles,movie_posters


def recommend_movie():
  movie_name = st.text_input('Movie Name: ')
  print(f'\nMovie Name: {movie_name}')
  movie_name = re.sub(r'\W+', '', movie_name)
  
  all_movie_Title = movie_dataframe['Title2'].tolist()
  get_close_match = difflib.get_close_matches(movie_name,all_movie_Title)
  
  if len(get_close_match) == 0:
    print(f'No close Match Found')
  else:
    print(f'\nClose Matches')
    close_match = get_close_match[0]
    for index,match in enumerate(get_close_match,start = 1):
      print(f'{index}. {match}')
    print(f'\nChoosen Movie name: {close_match}')


    movie_index = movie_dataframe[movie_dataframe.Title2 == close_match].index[0]
    print(f'Movie index: {movie_index}')

    similarity_score = list(enumerate(similarity[movie_index]))
    sorted_similarity_score = sorted(similarity_score,key = lambda x: x[1],reverse = True)

    print('\n Movie Suggested For You:')
    movie_titles,movie_posters = get_movie_posters(sorted_similarity_score,movie_dataframe)
    return movie_titles,movie_posters
    #show_movie_posters(movie_posters,movie_titles)
      

movie_titles,movie_posters = recommend_movie()
print(movie_posters)

# Divide the poster_urls list into chunks of 3
poster_chunks = [movie_posters[i:i+3] for i in range(0, len(movie_posters), 3)]
title_chunks = [movie_titles[i:i+3] for i in range(0, len(movie_titles), 3)]


for i in range(len(poster_chunks)):
    st.write("") # add some space before the row
    col1, col2, col3 = st.columns(3)
    with col1:
        if len(poster_chunks[i]) > 0:
            st.image(poster_chunks[i][0], width=250)
            st.markdown(f"<p style='text-align: center;'>{title_chunks[i][0]}</p>", unsafe_allow_html=True)  # add the movie title above the poster
    with col2:
        if len(poster_chunks[i]) > 1:
            st.image(poster_chunks[i][1], width=250)
            st.markdown(f"<p style='text-align: center;'>{title_chunks[i][1]}</p>", unsafe_allow_html=True)  # add the movie title above the poster
    with col3:
        if len(poster_chunks[i]) > 2:
            st.image(poster_chunks[i][2], width=250)
            st.markdown(f"<p style='text-align: center;'>{title_chunks[i][2]}</p>", unsafe_allow_html=True)  # add the movie title above the poster
