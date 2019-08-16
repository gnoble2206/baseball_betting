#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib

# loaded_model = joblib.load('saved_model.pkl')

def get_data():
    df = pd.read_csv('../data/team_batting/allbatting.csv')
    df1 = pd.read_csv('../data/team_pitching/allpitching.csv')

    merged_rows = df.merge(df1, how="inner", left_index=True, right_index=True)
    return merged_rows


def last_three(df, col_list):
    last_3 = lambda x: x.rolling(3).mean().shift(1)
    df[col_list] = df[col_list].apply(last_3)
    return df


col_list = ['PA', 'AB_x', 'R_x', 'H_x', '2B_x', '3B_x', 'HR_x', 'RBI', 'BB_x', 'IBB_x',
       'SO_x', 'HBP_x', 'SH_x', 'SF_x', 'ROE_x', 'GDP_x', 'SB_x', 'CS_x',
       'BA', 'OBP', 'SLG', 'OPS', 'LOB', '#_x', 'IP', 'H_y', 'R_y', 'ER', 'UER',
       'BB_y', 'SO_y', 'HR_y', 'HBP_y', 'ERA', 'BF', 'Pit', 'Str', 'IR',
       'IS', 'SB_y', 'CS_y', 'AB_y', '2B_y', '3B_y', 'IBB_y', 'SH_y',
       'SF_y', 'ROE_y', 'GDP_y', '#_y']


def compress_rows(df):
    df['date_month'] = df['Date_x'].str.split(' ', expand=True)[0]
    df['date_day'] = df['Date_x'].str.split(' ', expand=True)[1]
    months = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    df['date_month'] = df['date_month'].map(months)
    df['date_month'] = df['date_month'].astype(str)
    df['date_month'] = df['date_month'].str.zfill(2)
    df['date_day'] = df['date_day'].str.zfill(2)
    df[' Year_x'] = df[' Year_x'].astype(float)
    df[' Year_x'] = df[' Year_x'].astype(int)
    df[' Year_x'] = df[' Year_x'].astype(str)
    df['full_date'] = df[' Year_x'] + '-'+ df['date_month'] + '-' + df['date_day']
    df['full_date'] = df['full_date'].str.rstrip('susp')
    df['full_date'] = pd.to_datetime(df['full_date'])
    df.sort_values(by=['full_date', 'Umpire'], inplace=True)
    return df

team_codes = {
'LAD': '01', 'TBR': '02', 'MIL': '03', 'SEA': '04', 'TOR': '05', 'LAA': '06', 'OAK': '07','HOU': '08',
'KCR': '09', 'BOS': '10', 'PHI': '11', 'ARI': '12', 'CHC': '13', 'ATL': '14', 'SFG': '15', 'COL': '16',
'NYM': '17', 'SDP': '18', 'TEX': '19', 'MIN': '20', 'NYY': '21', 'WSN': '22', 'STL': '23', 'BAL': '24',
'PIT': '25', 'CIN': '26', 'CLE': '27', 'CHW': '28', 'MIA': '29',
'DET': '30'}


def combine_rows(df3):
    team1 = df3.iloc[::2].copy()
    team2 = df3.iloc[1::2].copy()
    game_count = [i for i in range(1,13639)]
    team1['game_id'] = game_count
    team2['game_id'] = game_count
    full_combined = pd.merge(team1, team2, on='game_id')
    return full_combined


# full_combined = combine_rows(merged_rows)

def final_combine(full_combined):
    full_combined['team_1'] = full_combined['Opp_x_x'].map(team_codes)
    full_combined['team_2'] = full_combined['Opp_x_y'].map(team_codes)
    full_combined['Unnamed: 2_x_y'] = full_combined['Unnamed: 2_x_y'].fillna(1)
    full_combined['Unnamed: 2_x_y'] = full_combined['Unnamed: 2_x_y'].map({'@': 0, 1: 1})
    df4 = pd.read_csv('../data/odds_cleaned.csv')
    full_combined['full_date_y'] = pd.to_datetime(full_combined['full_date_y'])
    full_combined['stats_join'] = full_combined['full_date_y'].apply(lambda x: x.strftime('%Y%m%d')) + full_combined['team_1'].astype(str) + full_combined['team_2'].astype(str)
    full_combined['Home Team'] = np.where(full_combined['Unnamed: 2_x_y'] == 0, full_combined['Opp_x_y'], full_combined['Opp_x_x'])
    full_combined['Away Team'] = np.where(full_combined['Unnamed: 2_x_y'] == 1, full_combined['Opp_x_y'], full_combined['Opp_x_x'])
    full_combined['Home Code'] = full_combined['Home Team'].map(team_codes)
    full_combined['Away Code'] = full_combined['Away Team'].map(team_codes)
    full_combined['stats_join'] = full_combined['full_date_y'].apply(lambda x: x.strftime('%Y%m%d')) + full_combined['Home Code'].astype(str) + full_combined['Away Code'].astype(str)
    stats_odds = pd.merge(full_combined, df4, on='stats_join')
    stats_odds['result'] = np.where(stats_odds['Final_x'] - stats_odds['Final_y'] > 0, 0, 1)
    stats_odds1 = stats_odds.copy()
    return stats_odds1

def main(clf, stats_odds1):
    stats_odds1.drop(columns=['Thr_x',
       'Opp. Starter (GmeSc)_x', ' Year_x_x', 'Gtm_y_x', 'Date_y_x',
       'Unnamed: 2_y_x', 'Opp_y_x', 'Rslt_y_x', 'IP_x', 'H_y_x', 'R_y_x',
       'ER_x', 'UER_x', 'BB_y_x', 'SO_y_x', 'HR_y_x', 'HBP_y_x', 'ERA_x',
       'BF_x', 'Pit_x', 'Str_x', 'IR_x', 'IS_x', 'SB_y_x', 'CS_y_x',
       'AB_y_x', '2B_y_x', '3B_y_x', 'IBB_y_x', 'SH_y_x', 'SF_y_x',
       'ROE_y_x', 'GDP_y_x', '#_y_x', 'Umpire_x',
       'Pitchers Used (Rest-GameScore-Dec)_x', ' Year_y_x',
       'date_month_x', 'date_day_x', 'full_date_x', 'game_id_x',
       'Gtm_x_y', 'Date_x_y', 'Unnamed: 2_x_y', 'Opp_x_y', 'Rslt_x_y', 'Thr_y',
       'Opp. Starter (GmeSc)_y', ' Year_x_y', 'Gtm_y_y', 'Date_y_y',
       'Unnamed: 2_y_y', 'Opp_y_y', 'Rslt_y_y', 'IP_y', 'H_y_y', 'R_y_y',
       'ER_y', 'UER_y', 'BB_y_y', 'SO_y_y', 'HR_y_y', 'HBP_y_y', 'ERA_y',
       'BF_y', 'Pit_y', 'Str_y', 'IR_y', 'IS_y', 'SB_y_y', 'CS_y_y',
       'AB_y_y', '2B_y_y', '3B_y_y', 'IBB_y_y', 'SH_y_y', 'SF_y_y',
       'ROE_y_y', 'GDP_y_y', '#_y_y', 'Umpire_y',
       'Pitchers Used (Rest-GameScore-Dec)_y', ' Year_y_y',
       'date_month_y', 'date_day_y', 'full_date_y', 'team_1', 'team_2',
       'stats_join', 'Home Team_x', 'Away Team', 'Home Code_x',
       'Away Code', 'Unnamed: 0', 'Unnamed: 0.1', 'game_id_y', 'date',
       'Visitor Team', 'Visitor Pitcher', 'Home Team_y', 'Home Pitcher', 'Home Code_y', 'Vis Code', 'Final_x', 'Final_y',
        'Gtm_x_x', 'Date_x_x', 'Unnamed: 2_x_x', 'Opp_x_x', 'Rslt_x_x'], inplace=True)

    stats_odds1_16 = stats_odds1.iloc[:7419]

    stats_odds1_17 = stats_odds1.iloc[7419:]

    stats_odds1_16 = stats_odds1_16.dropna()

    stats_odds1_17 = stats_odds1_17.dropna()
    # return stats_odds1_16, stats_odds1_17


    hopen_ml_16 = stats_odds1_16.pop('Home Open ML') 
    vopen_ml_16 = stats_odds1_16.pop('Open Visitor ML')
    stats_odds1_16.drop(columns=['Close Visitor ML', 'Home Close ML'], inplace=True)

    hopen_ml_17 = stats_odds1_17.pop('Home Open ML') 
    vopen_ml_17 = stats_odds1_17.pop('Open Visitor ML')
    stats_odds1_17.drop(columns=['Close Visitor ML', 'Home Close ML'], inplace=True)


    y = stats_odds1_16.pop('result')

    X = stats_odds1_16

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

    # clf = RandomForestClassifier(n_estimators=100, max_features=10, max_depth=10, n_jobs=-2)

    # clf.fit(X_train, y_train)

    y_preds = clf.predict(X_test)

    clf.predict_proba(X_test)

    train_accuracy = np.mean(cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy'))
    recall = np.mean(cross_val_score(clf, X_train, y_train, cv=5, scoring='recall'))
    precision = np.mean(cross_val_score(clf, X_train, y_train, cv=5, scoring='precision'))
    print('Random Forest Train: ')
    print('Train Accuracy: ', train_accuracy)
    print('Recall: ', recall)
    print('Precision: ', precision)

    list(zip(X, clf.feature_importances_))

    feature_import = list(zip(X, clf.feature_importances_))
    feature_import.sort(key=lambda x: x[1], reverse=True)
    feature_import = feature_import[:10]
    plt.bar(*zip(*feature_import))
    plt.xticks(rotation='vertical')

    ####Logistic Regression

    clf1 = LogisticRegression(solver='lbfgs', max_iter=2000)
    clf1.fit(X_train, y_train)
    y_preds1 = clf1.predict(X_test)
    train_accuracy1 = np.mean(cross_val_score(clf1, X_train, y_train, cv=5, scoring='accuracy'))
    recall1 = np.mean(cross_val_score(clf1, X_train, y_train, cv=5, scoring='recall'))
    precision1 = np.mean(cross_val_score(clf1, X_train, y_train, cv=5, scoring='precision'))
    print("Logistic Regression Train:")
    print("Train Accuracy: ", train_accuracy1)
    print("Recall: ", recall1)
    print("Precision: ", precision1)

    #### test results on holdout
    y = stats_odds1_17.pop('result')

    X = stats_odds1_17

    y_preds = clf.predict(X)
    probs = clf.predict_proba(X)
    y_pred1 = clf1.predict(X)

    print("Random Forest Holdout:")
    print("Holdout Accuracy:",metrics.accuracy_score(y, y_preds))
    print("Holdout Recall:", metrics.recall_score(y, y_preds))
    print("Holdout Precision:", metrics.precision_score(y, y_preds))

    print("Logistic Regression Holdout:")
    print("Train Accuracy: ",metrics.accuracy_score(y, y_pred1))
    print("Recall: ", metrics.recall_score(y, y_pred1))
    print("Precision: ", metrics.precision_score(y, y_pred1))


    X['prob_0'] = probs[:,0] 
    X['prob_1'] = probs[:,1]

    X['y_preds'] = y_preds

    X['result'] = y

    X['Open Visitor ML'] = vopen_ml_17 

    X['Home Open ML'] = hopen_ml_17

    X['Visitors Odds Prob'] = X['Open Visitor ML'].apply(lambda x: abs(x)/(abs(x) + 100) if x < 100 else 100/(x+100))
    X['Home Odds Prob'] = X['Home Open ML'].apply(lambda x: abs(x)/(abs(x) + 100) if x < 100 else 100/(x+100))

    X['max_model_prob'] = X[["prob_0", "prob_1"]].max(axis=1) * 100

    X['max_odds_prob'] = X[['Visitors Odds Prob', 'Home Odds Prob']].max(axis=1) * 100

    X['potential edge'] = X['max_model_prob'] - X['max_odds_prob']

    X['wager'] = X['potential edge'].apply(lambda x: 10 if x > 0 else 0)

    X['Home Payout'] = X['Home Open ML'].apply(lambda x: (100/abs(x) + 1) if x < 100 else (x/100 + 1))                               
    X['Visitor Payout'] = X['Open Visitor ML'].apply(lambda x: (100/abs(x) + 1) if x < 100 else (x/100 + 1))        
    X['Home Payout'] = X['Home Open ML'].apply(lambda x: (100/abs(x) + 1)*10 if x < 100 else (x/100 + 1)*10)                               
    X['Visitor Payout'] = X['Open Visitor ML'].apply(lambda x: (100/abs(x) + 1)*10 if x < 100 else (x/100 + 1)*10) 


    X['incorrect'] = np.where(X['result'] == X['y_preds'], 0, 1)

    conditions = [
        (X['incorrect'] == 0) & (X['y_preds'] == 0) & (X['wager'] == 10),
        (X['incorrect'] == 0) & (X['y_preds'] == 1) & (X['wager'] == 10),
        (X['incorrect'] == 1) & (X['wager'] == 10),
        (X['incorrect'] == 1) & (X['wager'] == 0)]
    choices = [X['Visitor Payout'], X['Home Payout'], 0, 0]
    X['payout'] = np.select(conditions, choices)

    X['payout'].sum()

    X['wager'].sum()

    roi = (X['payout'].sum() / X['wager'].sum() - 1) * 100

    print('ROI = {}'.format(roi))


if __name__ == '__main__':
    clf = joblib.load('saved_model.pkl')
    merged_rows = get_data()
    merged_rows = last_three(merged_rows, col_list)
    merged_rows.dropna(subset=['Date_x'], inplace=True)
    merged_rows = compress_rows(merged_rows)
    full_combined = combine_rows(merged_rows)
    stats_odds1 = final_combine(full_combined)
    main(clf, stats_odds1)







