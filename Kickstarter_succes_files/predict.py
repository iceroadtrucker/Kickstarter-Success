import pickle
from pathlib import Path
import pandas as pd
import typer
import glob
import os
from data_cleaning_feature_engineering import preprocessing


def load_ressources():
    with open("model/best_model_xgboost.sav", "rb") as model_xgboost:
        model = pickle.load(model_xgboost)
    return model


app = typer.Typer()

@app.command()
def predict(path):
    """
    Command line command that can be executed with a parameter
    """
    print('Loading the model...')
    model = load_ressources()
    # TODO write code to use model and return predictions
    
    print('Loading the project...')
    df = pd.read_csv(path)
    print('Preprocessing the data...')
    df = preprocessing(df)
    df = df.drop('state', axis=1)
    x_pred = df 
    print('Predicting the outcome...')
    y_pred = model.predict(x_pred)
    y_pred = y_pred.round(2)
    print('Predicting probabilities for success...')
    y_pred_proba = model.predict_proba(x_pred)
    y_pred_new = []
    for pred in y_pred:
        if pred == 1:
            y_pred_new.append('success')
        else:
            y_pred_new.append('failure')
    print('-------------------------------------------------------------------')
    print('The predicted outcome of your project:', y_pred_new)
    print('The predicted probability of your project being successful:', (y_pred_proba[:, 1] * 100).round(2), '%')
    print('Most important Features according to the columns:', model.feature_importances_.argsort())
    return 

path = input('Input the path of your data:')
predict(path)

#if __name__ == "__main__":
    #app()
