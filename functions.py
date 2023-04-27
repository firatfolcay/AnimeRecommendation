import pandas as pd

# Load anime dataframe

def getOneHot(animes_df):
    
    # Get unique genres
    unique_genres = sorted(set(g for genres in animes_df['genre'].fillna('') for g in genres.split(',')))

    # One-hot encode genres
    for value in unique_genres:
        animes_df[value] = animes_df['genre'].str.contains(value, case=False).fillna(False).astype(int)

    # Drop the original genre column
    animes_df = animes_df.drop('genre', axis=1)
    
    return animes_df
def csvToVector(dataframe):
    testframe = dataframe.loc[:, "Adventure":"Yaoi"]
    list_of_lists = [[row[col] for col in dataframe.loc[:, 'Action':'Yaoi'].columns] for _, row in dataframe.iterrows()]
    return list_of_lists

if __name__ == "__main__":

    # CHECK: if the csv file getting from the right place
    animes_df = getOneHot(pd.read_csv("/home/agirnob/Downloads/anime archive/anime.csv"))
    csvToVector(animes_df)
    # Save the encoded dataframe to a CSV file
    animes_df.to_csv('anime_encoded.csv', index=False)