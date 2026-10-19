from pathlib import Path
import pandas as pd
import sklearn
import sys


import pickle

if len(sys.argv) != 2:
    print("Usage: python Testing_script.py test.csv")
    sys.exit(1)

csv_file = sys.argv[1]

#Data_path = Path('datasets') / 'test.csv'
#csv_file = Path(__file__).parent / Data_path 
print(csv_file)
df = pd.read_csv(csv_file, sep=';')    #import data



Model_path = Path('model') / 'random_forest_model.pkl'
abs_model_path = Path(__file__).parent / Model_path 
with open(abs_model_path, 'rb') as file:
    model = pickle.load(file)               #import model



if "class" in df.columns:
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values 

    predictions = model.predict(X)
    corr_predict = sum(predictions == y)
    total_predict = len(y)

    accuracy = f"accuracy: { corr_predict} / {total_predict}"
    print(accuracy)   

else:
    
    X_test = df.values
    predictions = model.predict(X_test)  
    print(predictions)

