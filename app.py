# -*- coding: utf-8 -*-
"""

@author: Sergio Varela
"""


#1. Library Imports
import uvicorn
from fastapi import FastAPI
from FlowerClassifications import FlowerClassification 
import numpy as np
import pickle
import pandas as pd

# 2. Create the app object
app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

# 3. Index route, open automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, stranger'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere

@app.get('/docs')
def get_name(name: str):
    return {'Welcome to a Flower Classification API'}

# 5. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Flower Species

@app.post('/predict')
def predict_flower(data:FlowerClassification):
    data = data.dict()
    sepal_length=data['SepalLenghtCm']
    sepal_width=data['SepalWidthCm']
    petal_length=data['PetalLengthCm']
    petal_width=data['PetalWidthCm']
    #print(classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]]))
    prediction = classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    #print(prediction)
    if(prediction[0]==0):
        prediction = "Iris-setosa" 
        #print(prediction)
    elif(prediction[0]==1):
        prediction = "Iris-versicolor"
        #print(prediction)
    else:
        prediction = "Iris-virginica"
        #print(prediction)
    return {
         'prediction': prediction
         }
        
# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000

if __name__ == '__main__':
     uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload 

#6.4,3.2,5.3,2.3

