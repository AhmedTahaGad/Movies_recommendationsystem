# Movies_recommendationsystem
Predicting movie ratings using the K-Nearest Neighbors algorithm based on various features of movies.
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import re

with open('H:\ML\movies.dat', 'r', encoding='latin1') as file:
    data = file.readlines()
    #"movieID", "title", "year", "genres"
movie_data = []
for line in data:
    parts = re.split(r'::|\(', line.strip())
    ID = int(parts[0])
    title = parts[1].strip()
    year_match = re.search(r'\d{4}', parts[2])
    year = int(year_match.group()) if year_match else None
    genres = parts[3].rstrip(')').split("|")
    movie_data.append([ID, title, year, genres])
    
columns = ["ID", "title", "year", "genres"]
movies_df = pd.DataFrame(movie_data, columns=columns)

def process_year(year):
    if pd.notna(year):  
        year_str = str(int(year))
        return year_str.replace(".", "").lstrip("0")
    return None

movies_df['year'] = movies_df['year'].apply(process_year)

with open(r'H:\ML\users.dat', 'r', encoding='latin1') as file:
    data = file.readlines()
  #UserID::Gender::Age::Occupation::Zip-code 
users_data = []
for line in data:
    parts = re.split(r'::|\(', line.strip())
    ID = int(parts[0])
    Gender = parts[1].strip()
    Age = int(parts[2])
    Occupation = parts[3].strip()
    Zip_code = parts[4].strip()
    users_data.append([ID, Gender, Age, Occupation, Zip_code])
    
columns = ["ID", "Gender", "Age", "Occupation", "Zip-code"]
users_df = pd.DataFrame(users_data, columns=columns)

with open(r'H:\ML\ratings.dat', 'r', encoding='latin1') as file:
    data = file.readlines()
#UserID::MovieID::Rating::Timestamp
rating_data = []

for line in data:
    parts = re.split(r'::|\(', line.strip())
    UserID = int(parts[0])
    MovieID = int(parts[1])
    Rating = int(parts[2])
    Timestamp = parts[3].strip()
    
    rating_data.append([UserID, MovieID, Rating, Timestamp])
    
columns = ["UserID", "MovieID", "Rating", "Timestamp"]
rating_df = pd.DataFrame(rating_data, columns=columns)


Allgenres= ['Action', 'Adventure', 'Animation', "Children's" , 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
'Sci-Fi', 'Thriller', 'War', 'Western']

#movies_df

for genre in Allgenres:
    movies_df[genre] = 0

for index, row in movies_df.iterrows():
    for genre in row['genres']:
        if genre in Allgenres:
            movies_df.at[index, genre] = 1

#change_name = {"movieID": "MovieID"}
#movies_df.rename(columns=change_name, inplace=True)

user_genres_df = rating_df.merge(movies_df, left_on= 'MovieID', right_on = 'ID', how ='inner').drop('ID',axis=1)

user_genres_df =user_genres_df.drop(['title','MovieID','year','Timestamp'],axis=1)

user_genres_df.drop(columns=['genres'], inplace=True)

for genre in Allgenres:
    user_genres_df.loc[:, genre] = user_genres_df['Rating'] * user_genres_df[genre]

for col in user_genres_df.columns:
    user_genres_df[col+"_user_avg_Rating"] = user_genres_df[col]

user_genres_df["user_overall_avr_Rating"] = user_genres_df["Rating"]
user_genres_df = user_genres_df.drop("Rating",axis=1)
user_genres_df.head()

user_genres_df.columns.tolist()

user_genres_df = user_genres_df.drop("UserID_user_avg_Rating",axis=1)

user_genres_df = user_genres_df.drop(Allgenres, axis=1)

user_genres_df = user_genres_df.groupby('UserID').mean()

user_genres_df.reset_index(inplace=True)

final_df = rating_df.merge(user_genres_df, on='UserID', how='inner')

final_df = final_df.merge(movies_df, left_on='MovieID',right_on='ID', how='inner').drop('ID', axis=1)
final_df

final_df = final_df.drop(columns=['MovieID','Timestamp','title', 'year','genres',])

final_df.columns.tolist()

#KNN Model
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint
import pickle

X = final_df.drop('Rating', axis=1)
y = final_df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'n_neighbors': randint(1, 50),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto']
}

knn = KNeighborsClassifier()

randomized_search = RandomizedSearchCV(estimator=knn, param_distributions=param_dist, scoring='accuracy', cv=3, n_iter=5, error_score='raise')
randomized_search.fit(X_train, y_train)

best_knn_model = randomized_search.best_estimator_
best_params = randomized_search.best_params_

accuracy = best_knn_model.score(X_test, y_test)
print("Best Model Accuracy:", accuracy)
print("Best Hyperparameters:", best_params)


pickle.dump(best_knn_model, open("model_clf_pickle", 'wb'))

my_model_clf = pickle.load(open("model_clf_pickle", 'rb'))
result_score = my_model_clf.score(X_test, y_test)
print("Score:", result_score)

#0.96 ~ 96% accuracy
