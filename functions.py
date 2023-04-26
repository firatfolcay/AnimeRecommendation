import pandas as pd

# Load anime dataframe
animes_df = pd.read_csv('anime.csv')

# Get unique genres
unique_genres = sorted(set(g for genres in animes_df['genre'].fillna('') for g in genres.split(',')))

# One-hot encode genres
for value in unique_genres:
    animes_df[value] = animes_df['genre'].str.contains(value, case=False).fillna(False).astype(int)

# Drop the original genre column
animes_df = animes_df.drop('genre', axis=1)

# Save the encoded dataframe to a CSV file
animes_df.to_csv('anime_encoded.csv', index=False)
