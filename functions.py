import pandas as pd


# This Function generate unique genres and add to dataframe and assing 1 with corresponding genres
# input: dataframe from anime.csv
# output: dataframe with onehots encodings
def getOneHot(onehot_df):
    # Get unique genres
    unique_genres = sorted(set(g for genres in onehot_df['genre'].fillna('') for g in genres.split(',')))

    # One-hot encode genres
    for value in unique_genres:
        onehot_df[value] = onehot_df['genre'].str.contains(value, case=False).fillna(False).astype(int)

    # Drop the original genre column
    onehot_df = onehot_df.drop('genre', axis=1)

    return onehot_df

# This function converts dataframe to list of lists
# input: dataframe with one hot encodings
# output: listoflists of one-hot encoding
def csvToVector(dataframe):
    list_of_lists = [[row[col] for col in dataframe.loc[:, 'Action':'Yaoi'].columns] for _, row in dataframe.iterrows()]
    result_df = pd.DataFrame(list_of_lists, columns=dataframe.loc[:, 'Action':'Yaoi'].columns)
    result_list = multiByRating(dataframe, result_df)
    return result_list


# This Function multiplies one hot with rating scores then returns a matrix with that scores instead of 1s
# input: dataframe, dataframe
# output: dataframe
def multiByRating(dataframe, oneHotVector):
    ratings = dataframe['rating'].fillna(6.5)
    result_df = oneHotVector.mul(ratings, axis=0)
    return result_df.values.tolist()


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    # CHECK: if the csv file getting from the right place
    animes_df = getOneHot(pd.read_csv("anime.csv"))
    # oneHotVector = csvToVector(animes_df)
    print(csvToVector(animes_df))
    # Save the encoded dataframe to a CSV file
    animes_df.to_csv('anime_encoded.csv', index=False)
