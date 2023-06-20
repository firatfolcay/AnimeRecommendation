import pandas as pd
import numpy as np
import AnimeEncode
import StocasticReturns






if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    
    
    # CHECK: if the csv file getting from the right place
    
    #-----------------
    #GENRE BASED TEST|
    #-----------------
    encoder = AnimeEncode.GenreBased()
    animes_df = encoder.getOneHot(pd.read_csv("anime.csv"))

    oneHotVector = encoder.csvToVector(animes_df)
    npOneHot = np.array(oneHotVector)
    genreArray = encoder.getIndexDict(animes_df)

    ans = encoder.calcGenreBased([[[43], [62], [131], [327], [10991]]], npOneHot, 5)  # Örnek

    animes_df.loc[ans, 'name']
    choseList= StocasticReturns.calculate_combined_stochastic(animes_df, ans, 0.3, 0.7)
    animes_df.loc[choseList, 'name']
    
    #print(animes_df.loc[animes_df['anime_id'].isin(choseList)]['name'])
    # print(csvToVector(animes_df))
    # Save the encoded dataframe to a CSV file
    animes_df.to_csv('anime_encoded.csv', index=False)
    
    #-----------------
    # USER BASED TEST|
    #-----------------
    animes_df = pd.read_csv("anime.csv")
    users_df = pd.read_csv("rating.csv")
    
    animeList=['Mirai Nikki (TV)','Kuzu no Honkai','Highschool of the Dead','Ano Hi Mita Hana no Namae wo Bokutachi wa Mada Shiranai.','Angel Beats!','Boku wa Tomodachi ga Sukunai']
    animeListIndexes=list(animes_df.loc[animes_df['name'].isin(animeList)]['anime_id'])

    encoder = AnimeEncode.UserBased()
    userArray = encoder.getUserArray(users_df)
    userIndex = encoder.getUserIndex(users_df, animeListIndexes)   
    
    userId = encoder.findBestUserID()
    selected_ids = encoder.recommendByID(userId, users_df,20)
    print(animes_df.loc[animes_df['anime_id'].isin(selected_ids)]['name'])
    
    
    
    
    
    
