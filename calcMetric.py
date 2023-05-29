# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:48:08 2023

@author: firat
"""

import pandas as pd
import numpy as np
import AnimeEncode
import StocasticReturns
from sklearn.model_selection import train_test_split

#test_users_df = dataframe of selected test users
#user = user id of selected test user
def selectAnimeList(user, test_users_df):

    # Filter user's anime watched data
    user_anime_watched = test_users_df[test_users_df['user_id'] == user]['anime_id']
   
    
    # Randomly select half of the anime watched by the user
    half_num_anime = int(len(user_anime_watched) / 2)
    selected_anime = user_anime_watched.sample(n=half_num_anime, random_state=42)
  
    # Create a list of anime IDs for prediction then return it
    predicted_anime = len(user_anime_watched)-half_num_anime
    return selected_anime.tolist(), predicted_anime, user_anime_watched.tolist()
        


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    

    #---------------------------
    # PREPARE ENCODERS AND DATA|
    #---------------------------
    genreEncoder = AnimeEncode.GenreBased()
    userEncoder = AnimeEncode.UserBased()
    
    animes_df = genreEncoder.getOneHot(pd.read_csv("anime.csv"))
    users_df = pd.read_csv("rating.csv")
    
    userArray = userEncoder.getUserArray(users_df)
    
    #-----------------
    # PICK TEST USERS|
    #-----------------
     
    main_usr, test_usr = train_test_split(userArray, test_size=0.05, random_state=31)
    
    test_users_df = users_df.loc[users_df["user_id"].isin(test_usr)]
    
    main_users_df = users_df.loc[~users_df["user_id"].isin(test_usr)]

    
    #----------------------------------
    # CALCULATE USER BASED PERFORMANCE|
    #----------------------------------
    userScoresbyUserBased={}
    
    encoder = AnimeEncode.UserBased()
    userArray = encoder.getUserArray(users_df)
    
    for user in test_usr:
        animeList, predict_count, user_anime_watched = selectAnimeList(user,test_users_df)
        
        userIndex = userEncoder.getUserIndex(users_df, animeList)   
    
        userId = userEncoder.findBestUserID()
        predicted_animes = userEncoder.recommendByID(userId, users_df,predict_count)
        
        
        other_anime_ids = list(set(user_anime_watched) - set(animeList))

        
        correct_predictions = len(set(predicted_animes) & set(user_anime_watched))
        
        total_selected_anime = len(user_anime_watched)
        
        # Calculate the percentage of correctly predicted anime IDs
        percentage_correct = (correct_predictions / total_selected_anime) * 100

        userScoresbyUserBased[user] = percentage_correct        
        
        


    


    #-----------------------------------
    # CALCULATE GENRE BASED PERFORMANCE|
    #-----------------------------------
    
    oneHotVector = genreEncoder.csvToVector(animes_df)
    npOneHot = np.array(oneHotVector)
    genreArray = genreEncoder.getIndexDict(animes_df)
    
    userScoresbyGenreBased= {}
    
    for user in test_usr:
        animeList, predict_count, user_anime_watched = selectAnimeList(user,test_users_df)
        animeListIndices = animes_df.index[animes_df["anime_id"].isin(animeList)].tolist()
        
        predicted_animes_indices = genreEncoder.calcGenreBased([animeListIndices], npOneHot, predict_count)
        
        predicted_animes = animes_df.loc[predicted_animes_indices, 'anime_id'].tolist()
        
        other_anime_ids = list(set(user_anime_watched) - set(animeList))

        
        correct_predictions = len(set(predicted_animes) & set(user_anime_watched))
        
        total_selected_anime = len(user_anime_watched)
        
        # Calculate the percentage of correctly predicted anime IDs
        percentage_correct = (correct_predictions / total_selected_anime) * 100

        userScoresbyGenreBased[user] = percentage_correct        
        
    
    
    
    
    
    
    
