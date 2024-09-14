import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def predictor(dependent_variables):
    '''
    this function makes predictions using the trained 'loan approval prediction model'
     dependent_variable: a pandas series with the variables
      no_of_dependents,education,self_employed, income_annum, 
      loan_amount, loan_term, cibil_score, total_assets_value.
      eg: [5,0,1,16.097893,17.001863, 20,382,17.822844]
        '''
    regressor = joblib.load("loan_approval\loan_approval_linear_model_v1.pkl")
    prediction = regressor.predict(dependent_variables)
    return prediction