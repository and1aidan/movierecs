import pandas as pd

#read data from each csv file
movies_data = pd.read_csv('C:/Coding/movierec/movierecs/ml-latest-small/movies.csv')
ratings_data = pd.read_csv('C:/Coding/movierec/movierecs/ml-latest-small/ratings.csv')
tags_data = pd.read_csv('C:/Coding/movierec/movierecs/ml-latest-small/tags.csv')
#create new matrix using pivot_table for more data processing
ratings_agg = ratings_data.pivot_table(index='userId',columns='movieId',values='rating')

print(ratings_agg.shape)
