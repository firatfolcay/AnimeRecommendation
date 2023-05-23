# -*- coding: utf-8 -*-
"""
Created on Sat May 20 22:54:42 2023

@author: firat
"""

import pandas as pd
import numpy as np
import random
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
        self.userIndex=[0]*(len(self.userList))
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
        for user in user_df.iterrows():
            if user['user_id'] == userId:
                if  user['rating'] == -1:
                    user['rating'] = 5
                AnimeDict[user['anime_id']] = user['rating']
        
        total_rating = sum(AnimeDict.values())
        probabilities = {key: value / total_rating for key, value in user_df.items()}
        
        #selected_id = random.choices(list(probabilities.keys()), list(probabilities.values()))[0]
        selected_ids = random.choices(list(probabilities.keys()), list(probabilities.values()), k=count)
        
        return selected_ids

 
