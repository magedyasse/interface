
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    new_column_names = {
        'age': 'Age',
        'anaemia': 'Anaemia',
        'creatinine_phosphokinase': 'CreatininePhosphokinase',
        'diabetes': 'Diabetes',
        'ejection_fraction': 'EjectionFraction',
        'high_blood_pressure': 'HighBloodPressure',
        'platelets': 'PlateletCount',
        'serum_creatinine': 'SerumCreatinine',
        'serum_sodium': 'SerumSodium',
        'sex': 'Sex',
        'smoking': 'SmokingStatus',
        'time': 'FollowupDays',
        'DEATH_EVENT': 'DeathEvent'
    }
    df.rename(columns=new_column_names, inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def split_data(df, target_column='DeathEvent', features_to_drop=None):
    x = df.drop(columns=[target_column])
    y = df[target_column]
    if features_to_drop:
        x = x.drop(columns=features_to_drop)
    return train_test_split(x, y, stratify=y, random_state=42)

def resample_data(x_train, y_train):
    sm = SMOTEENN()
    X_resampled, y_resampled = sm.fit_resample(x_train, y_train)
    return X_resampled, y_resampled


