import numpy as np
import pandas as pd
import pathlib
import os
import time
import joblib
import multiprocessing

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


RANDOM_FOREST_SAVE_FILE = 'random_forest.joblib'
BACKUP_HASH_TO_TEAM_NAME = {}


def import_data(path): 
    return pd.read_csv(os.path.join(pathlib.Path().resolve(), 'data', 'raw', path), low_memory=False, )

def safe_dataframe(df: pd.DataFrame, filename):
    return df.to_csv(os.path.join(pathlib.Path().resolve(), 'data', 'final', filename), index=False)

def safe_randomForest(rf: RandomForestRegressor):
    joblib.dump(rf, os.path.join(pathlib.Path().resolve(), 'data', 'final', RANDOM_FOREST_SAVE_FILE))

def load_randomForest():
    return joblib.load(os.path.join(pathlib.Path().resolve(), 'data', 'final', RANDOM_FOREST_SAVE_FILE))

def get_prepared_dataframe() -> pd.DataFrame:
    '''
    Importiert die benötigten Daten und erstellt ein Dataframe, welches alle nebötigten Daten beinhaltet.
    '''
    raw_eco_dataframe = import_data('new\\economy.csv')
    raw_results_dataframe = import_data('new\\results.csv')
    #raw_player_dataframe = import_data('new\\players.csv')
    #raw_pick_dataframe = import_data('new\\picks.csv')
    
    # Copiert das Dataframe mit den benötigten Spalten
    eco_dataframe = raw_eco_dataframe[[ 'match_id', 'best_of']].copy()
    results_dataframe = raw_results_dataframe[['date','team_1','team_2','_map','result_1','result_2','map_winner','starting_ct', 'match_id','rank_1','rank_2','map_wins_1','map_wins_2','match_winner']].copy()

    # Verbinde beide Dataframes mit einem jopin über die match_id
    df = results_dataframe.join(eco_dataframe.set_index('match_id'), on='match_id')

    # Drop Rows mit best_of == 'o' oder fülle es mit 1 wenn es NaN ist
    # best_of (Ist eine angabe wie viele Spiele gespielt werden bis ein Sieger bestimmt ist. 
    # Z.b best_of 3 werden 3 Spiele gespielt und der Gewinner ist der, mit den meisten Siegen)
    df['best_of'] = df['best_of'].fillna(1)
    df.drop(df.loc[df['best_of'] == 'o'].index, inplace=True)    
    df['best_of'] = df['best_of'].astype(int)
    
    # date sollte vom datentyp datetime sein
    df['date'] = pd.to_datetime(df['date'])
    return df

def data_one_hot_encoding_maps(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Clear out default map wich not exists
    dataframe = dataframe.drop(dataframe[dataframe._map == 'Default'].index)

    # One hot encoding for each map-
    dataframe['dust'] = 0
    dataframe['inf'] = 0
    dataframe['vert'] = 0
    dataframe['train'] = 0
    dataframe['cobble'] = 0
    dataframe['mirage'] = 0
    dataframe['overpass'] = 0
    dataframe['cache'] = 0
    dataframe['nuke'] = 0

    for i, row in dataframe.iterrows():
            map = row['_map'].strip()
            if map != '':
                if map == 'Dust2':
                    dataframe.at[i,'dust'] = 1
                if map == 'Inferno':
                    dataframe.at[i,'inf'] = 1
                if map == 'Vertigo':
                    dataframe.at[i,'vert'] = 1
                if map == 'Train':
                    dataframe.at[i,'train'] = 1
                if map == 'Cobblestone':
                    dataframe.at[i,'cobble'] = 1
                if map == 'Mirage':
                    dataframe.at[i,'mirage'] = 1
                if map == 'Overpass':
                    dataframe.at[i,'overpass'] = 1
                if map == 'Cache':
                    dataframe.at[i,'cache'] = 1
                if map == 'Nuke':
                    dataframe.at[i,'nuke'] = 1

    dataframe.drop('_map', axis=1, inplace=True)

    return dataframe

def data_one_hot_encoding_date(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['year'] = 0
    dataframe['month'] = 0
    dataframe['day'] = 0

    for i, row in dataframe.iterrows():
        timestamp: pd.Timestamp = row['date']
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day

        # Update Date
        dataframe.at[i,'year'] = year
        dataframe.at[i,'month'] = month
        dataframe.at[i,'day'] = day

    dataframe.drop('date', axis=1, inplace=True)

    return dataframe

def integer_encode_teams(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Encoding teams (Integer Encoding)
    global BACKUP_HASH_TO_TEAM_NAME
    BACKUP_TEAM_NAMES = np.concatenate(
        (dataframe['team_1'].unique(), dataframe['team_2'].unique()), 
        axis=None
    )
    dataframe['team_1'] = dataframe['team_1'].apply(hash)
    dataframe['team_2'] = dataframe['team_2'].apply(hash)

    TEAM_HASHES = np.concatenate(
        (dataframe['team_1'].unique(), dataframe['team_2'].unique()), 
        axis=None
    )

    BACKUP_HASH_TO_TEAM_NAME = dict(zip(TEAM_HASHES, BACKUP_TEAM_NAMES))
    return dataframe

def data_one_hot_encoding(dataframe_raw: pd.DataFrame) -> pd.DataFrame:
    '''https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
    Random forest require all input variables and output variables to be numeric
    Convert dataframe to a numeric dataframe with encoding
    For categorical variables where no such ordinal relationship exists, the integer encoding is not enough
    one hot encode maps and date
    integer encode teams (hashing)
    '''
    dataframe = dataframe_raw.copy()
    
    dataframe = data_one_hot_encoding_maps(dataframe)
    dataframe = data_one_hot_encoding_date(dataframe)
    dataframe = integer_encode_teams(dataframe)
    
    return dataframe

def drop_duplicates_and_sort(dataframe: pd.DataFrame) -> pd.DataFrame:
    new_dataframe = dataframe.copy()
    # Remove all entires that exisit more than once
    new_dataframe = new_dataframe.drop_duplicates()

    # Sort dataframe in year month day, Random forest later will shuffle this data
    # just for better export and readability
    new_dataframe.sort_values(['year', 'month', 'day'], inplace=True)
    new_dataframe = new_dataframe.reset_index(drop=True)
    return new_dataframe

def train_random_forest(dataframe: pd.DataFrame):
    
    # Werte die Vorhergesagt werden sollen
    values_to_predict = np.array(dataframe['map_winner'])
    
    # Remove the labels from the features
    # axis 1 refers to the columns
    dataframebackup = dataframe
    dataframe = dataframe.drop('map_winner', axis = 1)
    dataframe = dataframe.drop('result_1', axis = 1)
    dataframe = dataframe.drop('result_2', axis = 1)
    dataframe = dataframe.drop('match_winner', axis = 1)

    # Convert to numpy array
    dataframe_array = np.array(dataframe)

    # Split the data into training and testing sets
    # test_size = all games played in 2020
    train_dataset, test_dataset, train_values, test_values = train_test_split(
        dataframe_array,
        values_to_predict,
        test_size=0.042975,
        random_state=66,
        shuffle=False
    )
    
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(
        n_estimators = 1000,
        max_depth=25,
        n_jobs=-1
    )

    # Train the model on training data and safe it to file
    rf.fit(train_dataset, train_values)
    #safe_randomForest(rf)

    # print most important features
    #importances = pd.DataFrame({'feature': dataframe.columns, 'importance':np.round(rf.feature_importances_,3)})
    #importances = importances.sort_values('importance',ascending=False).set_index('feature')
    #print(importances.head(20))
    
    return rf, test_dataset, train_values, dataframebackup, test_values

def predict_2020_games(rf, test_dataset, train_values, dataframebackup, test_values):
    # Create Predictions    
    predictions = rf.predict(test_dataset)

    # create new dataframe with predictions
    len_training = train_values.shape[0]
    array_test = np.zeros(len_training)
    prediction_values = np.concatenate((array_test, predictions), axis=None)
    dataframebackup['predictions'] = pd.Series(prediction_values)
    dataframebackup['predictions'] = dataframebackup['predictions'].round()
    dataframebackup['predictions'] = dataframebackup['predictions'].astype(int)


    # Revert team hash to name
    dataframebackup['team_1'] = dataframebackup['team_1'].apply(lambda x: BACKUP_HASH_TO_TEAM_NAME[x])
    dataframebackup['team_2'] = dataframebackup['team_2'].apply(lambda x: BACKUP_HASH_TO_TEAM_NAME[x])

    safe_dataframe(dataframebackup, 'result_prediction.csv')

    # Generate accuracy of predictions
    accuracy = 0
    # length of dataframe test values, ignore test data
    length = test_values.shape[0]
    for _, row in dataframebackup.iterrows():
        if (row['map_winner']  == row['predictions']):
            accuracy = accuracy + 1
    percentage = (accuracy / length) * 100
    print('Accuracy:', round(percentage, 2), '%.')


if __name__ == '__main__':
    # Zeit Messung
    t0 = time.process_time()

    dataframe = get_prepared_dataframe()
    dataframe = data_one_hot_encoding(dataframe)
    dataframe = drop_duplicates_and_sort(dataframe)

    t1 = time.process_time()
    
    print('Import and preperation done in: {}'.format(t1 - t0))
    
    safe_dataframe(dataframe, 'data.csv')

    random_forest, test_dataset, train_values, dataframebackup, test_values = train_random_forest(dataframe)
    
    t2 = time.process_time()
    print('Train random forest done in: {}'.format((t2 - t1) / multiprocessing.cpu_count()))

    predict_2020_games(random_forest, test_dataset, train_values, dataframebackup, test_values)

    t3 = time.process_time()
    print('Prediction done in: {}'.format(t3 - t2))


# Dataset
# https://www.kaggle.com/datasets/d3340095e128fb4e923b22b82b7ab97d134e40b9dea2d09e554d77a53ae7a768?resource=download


# Meaning of Column Names
# world rank: team world rank at the match date according to hltv.org rank.
# points: Number of points scored. It can be smaller than 5 if the match was a best of five game, in this case it would be possible to see t1points = 3 and t2points = 2, with t1 winning the confront, as an example.
# h2h win perc: float value from 0 to 1 representing share of wins on past games against the other team. Will be 0.5 if there is no previous confronts.
# player1 to player5 features: players are ordered from 1 to 5 according to their hltv.org rating(the best rated player will always be considered player1). Every single feature about the players are calculated over a three-month period before the match date and provided by hltv.org.
# rating: main metric of player quality.
# impact: overall impact of the player on games (usually correlated with important and decisive kills).
# kdr: total number of kills / total number of deaths.
# dmr: average damage per round.
# kpr: average number of kills per round.
# apr: average number of assists on teammates kills per round.
# dpr: average number of deaths per round.
# spr: average number of saved teammate per round.
# opk ratio: total opening kills on rounds / total opening deaths on rounds.
# opk rating: hltv score on the player opening kills.
# wins perc after fk: round conversion when the player get the first kill of the round.
# fk perc in wins: player first kill in won rounds.
# multikill perc: share of rounds where the player contributed with two or more kills.
# rating at least one perc: share of played matches where the player had at least 1.0 rating.
# is sniper: boolean indicating if the player has the "awp"(sniper) as his first or second weapon with most kills.
# clutch win perc: conversion rate from the player on one vs one situations.
