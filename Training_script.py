from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import pickle


Data_path = Path('datasets') / 'train.csv'
abs_data_path = Path(__file__).parent / Data_path 

df = pd.read_csv(abs_data_path, sep=';')

X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values   #posledni column = class

model=RandomForestClassifier()
model.fit(X,y)

Model_path = Path('model') / 'random_forest_model.pkl'
abs_model_path = Path(__file__).parent / Model_path

if not os.path.exists(abs_model_path.parent):
    os.makedirs(abs_model_path.parent)

with open(abs_model_path, 'wb') as file:
    pickle.dump(model, file)


