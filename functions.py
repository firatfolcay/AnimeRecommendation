from math import log

import pandas as pd
import numpy as np
import AnimeEncode
import StocasticReturns
from scipy.spatial.distance import cdist


# This Function applies genre based method input: npUserAnime is array list of indexes, npOneHotMatrix is our onehot
# list with scores, topXAnime is how much anime will be shown output: recommended anime indices
def calcGenreBased(npUserAnime, npOneHotMatrix, topXAnime):
    mean_point = np.mean(npOneHotMatrix[npUserAnime].reshape(len(npUserAnime[0]), 40), axis=0)
    print("mean point:" + "\n" + str(mean_point))
    distances = cdist(npOneHotMatrix, np.array([mean_point]), metric='euclidean')
    closest_point_indices = np.argsort(distances, axis=0)[:topXAnime, 0]
    # closest_points = npOneHotMatrix[closest_point_indices]
    return closest_point_indices







if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    # CHECK: if the csv file getting from the right place
    encoder = AnimeEncode.AnimeEncoder()
    animes_df = encoder.getOneHot(pd.read_csv("anime.csv"))

    oneHotVector = encoder.csvToVector(animes_df)
    npOneHot = np.array(oneHotVector)
    genreArray = encoder.getIndexDict(animes_df)

    ans = calcGenreBased([[[43], [62], [131], [327], [10991]]], npOneHot, 5)  # Örnek

    choseList= StocasticReturns.calculate_combined_stochastic(animes_df, ans, 0.3, 0.7)
    print(choseList)
    # print(csvToVector(animes_df))
    # Save the encoded dataframe to a CSV file
    animes_df.to_csv('anime_encoded.csv', index=False)
