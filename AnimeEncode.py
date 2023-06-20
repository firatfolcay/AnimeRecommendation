# -*- coding: utf-8 -*-
"""
Created on Sat May 20 22:54:42 2023

@author: firat
"""

import pandas as pd
import random
import numpy as np
from scipy.spatial.distance import cdist
#Anime Encoder for Genre Based
class GenreBased:
    def __init__(self):
        self.genreArray = []

    # This Function generate unique genres and add to dataframe and assing 1 with corresponding genres
    # input: dataframe from anime.csv
    # output: dataframe with onehots encodings
    def getOneHot(self, onehot_df):
        unique_genres = sorted(set(g for genres in onehot_df['genre'].fillna('') for g in genres.split(',')))
        for value in unique_genres:
            onehot_df[value] = onehot_df['genre'].str.contains(value, case=False).fillna(False).astype(int)
        onehot_df = onehot_df.drop('genre', axis=1)
        return onehot_df

    # This function converts dataframe to list of lists
    # input: dataframe with one hot encodings
    # output: listoflists of one-hot encoding
    def csvToVector(self, dataframe):
        list_of_lists = [[row[col] for col in dataframe.loc[:, 'Action':'Yaoi'].columns] for _, row in dataframe.iterrows()]
        result_df = pd.DataFrame(list_of_lists, columns=dataframe.loc[:, 'Action':'Yaoi'].columns)
        result_list = self.multiByRating(dataframe, result_df)
        return result_list

    # This Function multiplies one hot with rating scores then returns a matrix with that scores instead of 1s
    # input: dataframe, dataframe
    # output: dataframe
    def multiByRating(self, dataframe, oneHotVector):
        ratings = dataframe['rating'].fillna(6.5)
        result_df = oneHotVector.mul(ratings, axis=0)
        return result_df.values.tolist()

    # This Function is for determining genre in later steps because we coded genre alphabetically rather than given order
    # input: dataframe with one hot encodings
    # output: alphabetically indexed genre array
    def getIndexDict(self, dataframe):
        self.genreArray = [genre for genre in dataframe.loc[:, 'Action':'Yaoi'].columns]
        return self.genreArray
    
    # This Function applies genre based method input: npUserAnime is array list of indexes, npOneHotMatrix is our onehot
    # list with scores, topXAnime is how much anime will be shown output: recommended anime indices
    def calcGenreBased(self, npUserAnime, npOneHotMatrix, topXAnime):
        mean_point = np.mean(npOneHotMatrix[npUserAnime].reshape(len(npUserAnime[0]), 40), axis=0)
        print("mean point:" + "\n" + str(mean_point))
        distances = cdist(npOneHotMatrix, np.array([mean_point]), metric='euclidean')
        closest_point_indices = np.argsort(distances, axis=0)[:topXAnime, 0]
        # closest_points = npOneHotMatrix[closest_point_indices]
        return closest_point_indices

    
    def calculate_combined_deterministic(dataframe_df, closestpoints, topX, ws=0.7, wp=0.3):
        score = dataframe_df.loc[closestpoints, 'rating']
        popularity = dataframe_df.loc[closestpoints, 'members']
        combined_value = (score / 10 * ws) + (np.log(np.log(popularity.iloc[0])) * wp)
        chosen_element = combined_value.nlargest(topX).index.tolist()
        return chosen_element

        
""" 
#Anime encoder for User Based
class UserBased:
    def __init__(self):
        self.userList= []
        self.userIndex= []
    #This Functions creates an array that contains all user indexes
    #input: User data frame
    #output: An array that contains all user indexes
    def getUserArray(self, user_df):
        self.userList= user_df['user_id'].unique()
        
        return self.userList
    
    
    def getUserIndex(self, user_df,animeList):
        self.userIndex=[0]*(len(self.userList)+2)
        for i ,row in user_df.iterrows():
            
            if row['anime_id'] in animeList:
                
                if row['rating'] == -1:
                    
                    self.userIndex[row['user_id']] +=  5
                else: 
                    
                    self.userIndex[row['user_id']] += row[2]
                    
        
        return self.userIndex
    
    
    
    def findBestUserID(self):
        bestUserID = self.userIndex.index(max(self.userIndex))
        return bestUserID
    
    
    def recommendByID(self, userId, user_df, count=5):
        AnimeDict = {}
        for _, user in user_df.iterrows():
            if user['user_id'] == userId:
                if user['rating'] == -1:
                    user['rating'] = 5
                AnimeDict[user['anime_id']] = user['rating']
    
        total_rating = sum(AnimeDict.values())
        probabilities = {key: value / total_rating for key, value in AnimeDict.items()}
    
        selected_ids = random.choices(list(probabilities.keys()), list(probabilities.values()), k=count)
    
        return selected_ids
    
    
    def recommendByIDDeter(self, userId, user_df, count=5):
        AnimeDict = {}
        for _, user in user_df.iterrows():
            if user['user_id'] == userId:
                if user['rating'] == -1:
                    user['rating'] = 5
                AnimeDict[user['anime_id']] = user['rating']
    
        selected_ids= dict(list(sorted(AnimeDict).items())[0:count]) 
        return selected_ids
    
"""

class UserBased:
    def __init__(self):
        self.userList = []
        self.userIndex = []
    
    def getUserArray(self, user_df):
        self.userList = user_df['user_id'].unique()
        return self.userList
    
    def getUserIndex(self, user_df, animeList):
        self.userIndex = [0] * (len(self.userList) + 2)
        anime_ratings = user_df[user_df['anime_id'].isin(animeList)]
        
        for _, row in anime_ratings.iterrows():
            if row['rating'] == -1:
                self.userIndex[row['user_id']] += 5
            else:
                self.userIndex[row['user_id']] += row['rating']
        
        return self.userIndex
    
    def findBestUserID(self):
        bestUserID = self.userIndex.index(max(self.userIndex))
        return bestUserID
    
    def recommendByID(self, userId, user_df, count=5):
        anime_ratings = user_df[user_df['user_id'] == userId]
        anime_ratings = anime_ratings[anime_ratings['rating'] != -1]
        
        selected_ids = anime_ratings.sample(n=count, replace=True)['anime_id'].tolist()
        return selected_ids
    
    def recommendByIDDeter(self, userId, user_df, count=5):
        anime_ratings = user_df[user_df['user_id'] == userId]
        anime_ratings = anime_ratings[anime_ratings['rating'] != -1]
        
        selected_ids = anime_ratings.sort_values('anime_id').head(count)['anime_id'].tolist()
        return selected_ids

if __name__ == "__main__":
    animes_df = pd.read_csv("anime.csv")
    animeList=['Pumpkin Scissors','Ginga Eiyuu Densetsu','Shakugan no Shana','So Ra No Wo To','Gosick','Kingdom','Arslan Senki','Grisaia no Rakuen', 'Youjo Senki']
    animeListIndexes=list(animes_df.loc[animes_df['name'].isin(animeList)]['anime_id'])      
    animeList=[7 ,266 ,278 ,435,986,1300,2495,3286,11107]
    users_df = pd.read_csv("rating.csv")
    userBased=UserBased()
    c=userBased.getUserArray(users_df)
    d=userBased.getUserIndex(users_df, animeListIndexes)
    userId=userBased.findBestUserID()
    selected_ids=userBased.recommendByIDDeter(userId, users_df,10)
    print(animes_df.loc[animes_df['anime_id'].isin(selected_ids)]['name'])


    
    
    

