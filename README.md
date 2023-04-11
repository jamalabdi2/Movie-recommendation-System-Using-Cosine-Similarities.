# Movie Recommendation System using cosine similarity

This project aims to create a movie recommendation system using the content-based approach. Given a user's favorite movie, the system recommends similar movies based on the movie's attributes.


# This project uses the following libraries:

 1. Pandas
 2. Python
 3. difflib
 4. Scikit-learn
 5. Streamlit
 6. FastAPI
 7. Requests
 8. BeautifulSoup
 9. Matplotlib
 10. Seaborn

# Dataset
The dataset was scrapped from IMDb website using requests and beautifulSoup. Dataset was cleaned using regular expression and Pandas. 

# Feature Selection
The system uses the following attributes to recommend similar movies:

Genre
Director
Stars
Rating
Year
Description

# Data Visualization

Before performing feature selection, some data visualization was done to gain insights into the dataset. 
The following plots were generated:

Scatter plot of votes versus rating
Distribution plot of ratings
Bar plot of top 10 highest grossing movies
Scatter plot of metascores versus ratings
Bar plot of movie certificates

# Recommendation System

The recommendation system works in the following way:

1. Combine the selected attributes of each movie into a single string.
2. Convert the strings to a sparse matrix of TF-IDF vectors.
3. Compute the cosine similarity matrix of the vectors.
4. Ask the user for their favorite movie and preprocess the input.
5. Find the closest match for the input movie name in the dataset.
6. Display the top 20 most similar movies and their similarity scores.
7. To run the code, open it in an environment that supports Python and the required libraries. Make sure to have the movies.csv file in the same directory as the code file. The user will be prompted to enter their favorite movie name to receive movie recommendations.

# Recommendations

To generate recommendations, the user is prompted to enter the name of a movie. The system then finds the closest match to the movie title in the dataset using the get_close_matches function from the difflib module. The cosine similarity scores between the selected movie and all other movies in the dataset are then calculated. The top 12 movies with the highest similarity scores are selected and their posters are displayed.

# Credits

The dataset used in this project was obtained from the IMDb website.

