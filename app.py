from fastapi import FastAPI
from BankNotes import BankNote
import pandas as pd
import pickle
import uvicorn
import numpy as np

app=FastAPI(title="AI APP",
    summary="Deadpool's favorite app. Nuff said.",
    version="0.0.1",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "Vivek Nag Kanuri",
        "url": "https://x.com/VivekNagKanuri",
        "email":"viveknagkanuri@gmail.com",
    },
    )
pickle_in=open("rf.pkl","rb")
rf=pickle.load(pickle_in)
@app.get('/')
def index():
    return "Hello, User"

@app.get('/{name}')
def get_name(name:str):
    return f"Hello, {name}"

@app.post('/predict')
def predict_species(data:BankNote):
    data=data.dict()
    print(data)
    print("HELLO")
    variance=data['variance']
    print(variance)
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    print(rf.predict([[variance,skewness,curtosis,entropy]]))
    predictor=rf.predict([[variance,skewness,curtosis,entropy]])
    if(predictor[0]>0.5):
        return "Fake Note"
    else:
        return "Bank Note"
    return {
        'prediction':prediction
    }
if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)
          
