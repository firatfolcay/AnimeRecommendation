# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:48:08 2023

@author: firat
"""

import pandas as pd
import numpy as np
import AnimeEncode
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import seaborn as sns

#test_users_df = dataframe of selected test users
#user = user id of selected test user
def selectAnimeList(user, test_users_df):

    # Filter user's anime watched data
    user_anime_watched = test_users_df[test_users_df['user_id'] == user]['anime_id']
   
    
    # Randomly select half of the anime watched by the user
    half_num_anime = int(len(user_anime_watched) / 2)
    selected_anime = user_anime_watched.sample(n=half_num_anime, random_state=42)

    # Create a list of anime IDs for prediction then return it

    return selected_anime.tolist(), user_anime_watched.tolist()
        
def calculate_combined_deterministic(dataframe_df, closestpoints, topX, ws=0.7, wp=0.3):
    score = dataframe_df.loc[closestpoints, 'rating']
    popularity = dataframe_df.loc[closestpoints, 'members']
    combined_value = (score / 10 * ws) + (np.log(np.log(popularity.iloc[0])) * wp)
    chosen_element = combined_value.nlargest(topX).index.tolist()
    return chosen_element

def calculate_score_only(dataframe_df, closestpoints, topX, ws=0.7, wp=0.3):
    score = dataframe_df.loc[closestpoints, 'rating']
    chosen_element = score.nlargest(topX).index.tolist()
    return chosen_element

def process_user(user):
    try:
        animeList, user_anime_watched = selectAnimeList(user, test_users_df)
        predict_count = 20
        userIndex = userEncoder.getUserIndex(users_df, animeList)
        userId = userEncoder.findBestUserID()
        predicted_animes = userEncoder.recommendByID(userId, users_df, predict_count)
        other_anime_ids = list(set(user_anime_watched) - set(animeList))
    
        true_positives = len(set(predicted_animes) & set(user_anime_watched))
        false_positives = abs(predict_count - true_positives)
    
        precision = true_positives / (true_positives + false_positives)
        recall = 1 
        f1_score = 2 * ((precision * recall) / (precision + recall))

        return {'Anime Watched': len(user_anime_watched), 'Precision': precision, 'Recall': recall, 'F1 Score': f1_score}
    except Exception as e:
        print(f"Exception occurred for user {user}: {str(e)}")

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
    
    # Filter test users who have watched 20 or more anime
    filtered_test_users = []
    for user in test_usr:
        user_anime_watched = test_users_df[test_users_df['user_id'] == user]['anime_id']
        if len(user_anime_watched) >= 20:
            filtered_test_users.append(user)
    
    test_usr = filtered_test_users
    test_users_df = test_users_df[test_users_df["user_id"].isin(test_usr)]
    
    main_usr = list(set(main_usr) | set(test_usr))
    main_users_df = users_df.loc[users_df["user_id"].isin(main_usr)]
    
    #----------------------------------
    # CALCULATE USER BASED PERFORMANCE|
    #----------------------------------
    """
    encoder = AnimeEncode.UserBased()
    userArray = encoder.getUserArray(users_df)
    
    metrics = {'Precision': [], 'Recall': [], 'F1 Score': [], 'Accuracy': []}
    
    for user in test_usr:
        animeList, predict_count, user_anime_watched = selectAnimeList(user, test_users_df)
        predict_count=20
        userIndex = userEncoder.getUserIndex(users_df, animeList)
        userId = userEncoder.findBestUserID()
        predicted_animes = userEncoder.recommendByID(userId, users_df, predict_count)
        other_anime_ids = list(set(user_anime_watched) - set(animeList))
        
        true_positives = len(set(predicted_animes) & set(user_anime_watched))
        false_positives = len(set(predicted_animes) - set(user_anime_watched))
        false_negatives = len(set(user_anime_watched) - set(predicted_animes))
        true_negatives = len(other_anime_ids)
        
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        
        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)
        metrics['F1 Score'].append(f1_score)
        metrics['Accuracy'].append(accuracy)
    
    average_metrics = {metric: sum(metrics[metric]) / len(metrics[metric]) for metric in metrics}
    
    userScoresbyUserBased = {user: {'Precision': metrics['Precision'][i],
                                    'Recall': metrics['Recall'][i],
                                    'F1 Score': metrics['F1 Score'][i],
                                    'Accuracy': metrics['Accuracy'][i]}
                             for i, user in enumerate(test_usr)}
    
    """
    

    # Process users in parallel
    metrics = Parallel(n_jobs=-1, verbose=10)(delayed(process_user)(user) for user in test_usr)

    
    # Extract individual metric values
    anime_watched_values = [metric['Anime Watched'] for metric in metrics if metric is not None]
    precision_values = [metric['Precision'] for metric in metrics if metric is not None]
    recall_values = [metric['Recall'] for metric in metrics if metric is not None]
    f1_score_values = [metric['F1 Score'] for metric in metrics if metric is not None]
    
    # Remove None values from metrics
    metrics_wout_none = [metric for metric in metrics if metric is not None]

    
    # Create DataFrame from metrics
    df = pd.DataFrame(metrics_wout_none)
        
    # Remove extreme ends by setting a threshold for anime watched counts
    threshold = df['Anime Watched'].quantile(0.95)
    df = df[df['Anime Watched'] <= threshold]
    

    # Plot the correlations
    sns.regplot(data=df, x='Anime Watched', y='Precision', scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    plt.xlabel('Anime Watched')
    plt.ylabel('Precision')
    
    # Calculate the Pearson correlation coefficient
    corr, _ = pearsonr(df['Anime Watched'], df['Precision'])
    print("Pearson correlation coefficient:", corr)
    plt.title('Correlation between Watched Anime Counts and Precision (Pearson Correlation: {:.3f})'.format(corr))
    
    plt.show()


    """
    import pickle

    # Save metrics to a file
    with open('metrics.pkl', 'wb') as file:
        pickle.dump(metrics, file)

    # Load metrics from the file
    with open('metrics.pkl', 'rb') as file:
        loaded_metrics = pickle.load(file)
    """
    #-----------------------------------
    # CALCULATE GENRE BASED PERFORMANCE|
    #-----------------------------------
    
    oneHotVector = genreEncoder.csvToVector(animes_df)
    npOneHot = np.array(oneHotVector)
    genreArray = genreEncoder.getIndexDict(animes_df)
    """
    userScoresbyGenreBased= {}
    
    for user in test_usr:
        animeList, predict_count, user_anime_watched = selectAnimeList(user,test_users_df)
        animeListIndices = animes_df.index[animes_df["anime_id"].isin(animeList)].tolist()
        
        closest_points = genreEncoder.calcGenreBased([animeListIndices], npOneHot, 50)
        
        predicted_animes_indices = calculate_score_only(animes_df, closest_points, 10)
        
        predicted_animes = animes_df.loc[predicted_animes_indices, 'anime_id'].tolist()
        
        other_anime_ids = list(set(user_anime_watched) - set(animeList))

        
        correct_predictions = len(set(predicted_animes) & set(user_anime_watched))
        
        total_selected_anime = len(user_anime_watched)
        
        # Calculate the percentage of correctly predicted anime IDs
        percentage_correct = (correct_predictions / total_selected_anime) * 100

        userScoresbyGenreBased[user] = percentage_correct        
        
    """    
    # Initialize empty lists to store the data
    watched_counts = []
    precisions = []
    
    for user in test_usr:
        animeList, user_anime_watched = selectAnimeList(user, test_users_df)
        animeListIndices = animes_df.index[animes_df["anime_id"].isin(animeList)].tolist()
    
        closest_points = genreEncoder.calcGenreBased([animeListIndices], npOneHot, 50)
    
        predicted_animes_indices = calculate_score_only(animes_df, closest_points, 20)
    
        predicted_animes = animes_df.loc[predicted_animes_indices, 'anime_id'].tolist()
    
        other_anime_ids = list(set(user_anime_watched) - set(animeList))
    
        # Convert the predicted anime list to a binary classification problem
        predicted_positive = set(predicted_animes) & set(user_anime_watched)
        predicted_negative = set(other_anime_ids) - set(predicted_animes)
    
        # Calculate TP and FP
        true_positives = len(predicted_positive)
        false_positives = len(predicted_negative)
    
        # Calculate precision
        precision = true_positives / (true_positives + false_positives)
    
        # Append the data to the lists
        watched_counts.append(len(user_anime_watched))
        precisions.append(precision)
       
    
    
    # Create a DataFrame from the lists
    data = pd.DataFrame({'Watched Counts': watched_counts, 'Precision': precisions})
    
    # Remove extreme ends by setting a threshold for watched counts
    threshold = data['Watched Counts'].quantile(0.95)
    data = data[data['Watched Counts'] <= threshold]
    
    # Calculate the Pearson correlation coefficient
    corr, _ = pearsonr(data['Watched Counts'], data['Precision'])
    print("Pearson correlation coefficient:", corr)
    
    # Plot the correlations with regression line
    plt.scatter(data['Watched Counts'], data['Precision'], alpha=0.5)
    plt.xlabel('Watched Anime Counts')
    plt.ylabel('Precision')
    plt.title('Correlation between Watched Anime Counts and Precision (Pearson Correlation: {:.3f})'.format(corr))
    
    # Fit a regression line
    coefficients = np.polyfit(data['Watched Counts'], data['Precision'], 1)
    poly = np.poly1d(coefficients)
    x = np.linspace(data['Watched Counts'].min(), data['Watched Counts'].max(), 100)
    plt.plot(x, poly(x), color='red', label='Regression Line')
    
    plt.legend()
    plt.show()
        
    
    
    """
    plt.scatter(range(len(userScoresbyGenreBased)),list(userScoresbyGenreBased.values()), alpha=0.5)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot with Low Alpha')
    
    # Show the plot
    plt.show()
    
    count_zeros = sum(1 for value in userScoresbyGenreBased.values() if value == 0.0)
    print("Number of 0.0s:", count_zeros)
    
    count_hundered = sum(1 for value in userScoresbyGenreBased.values() if value == 100.0)
    print("Number of 100.0s:", count_hundered)
    
    """
    #---------------------------
    #         PLOTTING         |
    #---------------------------
    
    """


    # Define the values (Replace with your own values)
    true_positives = 85
    false_positives = 15
    true_negatives = 800
    false_negatives = 100
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    
    # Create the bar plot
    metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    values = [precision, recall, f1_score, accuracy]
    
    plt.bar(metrics, values)
    plt.ylim(0, 1)  # Set the y-axis limit to range from 0 to 1
    plt.ylabel('Value')
    plt.title('Performance Metrics')
    
    # Add text labels for each bar
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.show()
    
    
    """
    
