import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

#read data from each csv file
movies_data = pd.read_csv('C:/Coding/movierec/movierecs/ml-latest-small/movies.csv')
ratings_data = pd.read_csv('C:/Coding/movierec/movierecs/ml-latest-small/ratings.csv')
tags_data = pd.read_csv('C:/Coding/movierec/movierecs/ml-latest-small/tags.csv')
#create new matrix using pivot_table for more data processing
ratings_agg = ratings_data.pivot_table(index='userId',columns='movieId',values='rating')
#fill empty values with 0 instead of NaN
ra_filled = ratings_agg.fillna(0)

#verification
#print(ra_filled.head)
#print(ratings_agg.shape) #610, 9724

k = 50
svd = TruncatedSVD(n_components=k, random_state=19)

svd.fit(ra_filled)
u = svd.transform(ra_filled) # shape (n_users, k)
sigma = svd.singular_values_  # length k
v_t = svd.components_ # shape (k, n_items)

ra_hat = np.dot(u, np.diag(sigma).dot(v_t)) # reconstruct R = u * sigma * v_t (dot product)

def recommend(user_id, ra_filled, ra_reconstructed, movies_data, N=10):
    uidx = user_id = 1

    seen = set(np.where(ra_filled.iloc[uidx] > 0)[0])

    scores = list(enumerate(ra_reconstructed[uidx]))

    scores = [(mid, score) for mid, score in scores if mid not in seen]

    top = sorted(scores, key=lambda x: x[1], reverse=True)[:N]
    movie_ids = [ra_filled.columns[mid] for mid, _ in top]
    return movies_data.set_index('movieId').loc[movie_ids]['title']

print(recommend(1, ra_filled, ra_hat, movies_data, N=10))