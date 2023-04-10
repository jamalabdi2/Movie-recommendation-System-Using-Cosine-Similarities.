from fastapi import FastAPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import re
import difflib
from PIL import Image
import requests
from io import BytesIO


app = FastAPI()

# Load data from the JSON file
movie_dataframe = pd.read_json('/Users/jamal/Golang/cleaned_movie_dataframe.json')

@app.get("/movie-recommendation/{movie_name}")
async def movie_recommendation(movie_name: str):
    movie_titles = movie_dataframe['Title'].tolist()

    # Find the closest match of the input movie name in the dataset
    find_close_match = difflib.get_close_matches(movie_name, movie_titles)
    close_match = find_close_match[0]
    movie_location = movie_dataframe[movie_dataframe.Title == close_match]
    movie_index = movie_location.index[0]

    # Calculate the similarity score of all movies in the dataset
    def combine_features(movies):
        combined_features = ""
        for column in movies.columns:
            combined_features += movies[column].fillna('').astype(str) + ' '
        return combined_features.str.strip()

    def get_feature_vector(data):
        vectorizer = TfidfVectorizer()
        feature_vector = vectorizer.fit_transform(data)
        return feature_vector

    def get_cosine_similarity_matrix(feature_vector):
        similarity_matrix = cosine_similarity(feature_vector)
        return similarity_matrix

    selected_columns = ['Genre', 'Director', 'Stars', 'Rating', 'Year', 'Description']
    selected_features = movie_dataframe[selected_columns]
    def join_strings(df, column_name):
        df.loc[:, column_name] = df[column_name].apply(lambda x: ' '.join(x))
        return df

    join_strings(selected_features, 'Genre')
    join_strings(selected_features, 'Stars')

    combined_features = combine_features(selected_features)
    feature_vector = get_feature_vector(combined_features)
    similarity = get_cosine_similarity_matrix(feature_vector)

    similarity_score = list(enumerate(similarity[movie_index]))
    sort_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Retrieve the movie posters of top 12 similar movies
    def get_movie_posters(similar_movies, movie_dataframe):
        movie_posters = []
        movie_titles = []
        for movie in similar_movies[:12]:
            index = movie[0]
            movie_title = movie_dataframe.iloc[index].Title
            movie_poster = movie_dataframe.iloc[index]['Movie Poster']
            movie_posters.append(movie_poster)
            movie_titles.append(movie_title)

        return movie_posters, movie_titles


    similar_movies = get_movie_posters(sort_similar_movies, movie_dataframe)
    movie_posters = similar_movies[0]
    movie_titles = similar_movies[1]

    return {"movie_titles": movie_titles, "movie_posters": movie_posters}

