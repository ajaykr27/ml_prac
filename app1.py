# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI, Depends,HTTPException,Request,Form,status
from BankNotes import BankNote
import streamlit as st
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import numpy as np
from fastapi.responses import HTMLResponse
import pickle
import Models
import pandas as pd
from Database import engine, SessionLocal
from sqlalchemy.orm import Session

Models.Base.metadata.create_all(bind=engine)

templates = Jinja2Templates(directory="templates")

def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


# 2. Create the app object
app = FastAPI()
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

Notes=[]

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def home():
    return 'Welcome to the Note Prediction '

@app.get('/predictions')
def index(db: Session = Depends(get_db)):
    return db.query(Models.Notes).all()

@app.get('/predictions/{id}')
def prediction_table(id:int,db:Session=Depends(get_db)):
    predicted=db.query(Models.Notes).filter(Models.Notes.Id==id).first()
    if not predicted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Id Not Found")
    return predicted

@app.get("/predict/form", response_class=HTMLResponse)
async def read_item(request: Request):
    variance = ""
    skewness = ""
    curtosis = ""
    entropy = ""
    prediction=""
    return templates.TemplateResponse('predict.html',context={'request':request,'variance':variance,'skewness':skewness,'curtosis':curtosis,'entropy':entropy,'prediction':prediction})

"""@app.get("/predict/form")
def form_post(request: Request,db: Session = Depends(get_db), variance: float = Form(...),skewness:float=Form(...),curtosis:float=Form(...),entropy:float=Form(...)):
    predictions = index(db=db)
    variance=variance
    skewness=skewness
    curtosis=curtosis
    entropy=entropy

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return templates.TemplateResponse('predict.html',context={'request':request,'variance':variance,'skewness':skewness,'curtosis':curtosis,'entropy':entropy,'prediction':prediction})"""

@app.post('/predict',status_code=status.HTTP_201_CREATED)
async def add(request:BankNote,db:Session=Depends(get_db)):
    prediction=classifier.predict([[
        request.variance,
        request.skewness,
        request.curtosis,
        request.entropy,
    ]])
    request.prediction=prediction[0]
    new_prediction = Models.Notes(variance=request.variance,skewness=request.skewness,curtosis=request.curtosis,entropy=request.entropy,prediction=request.prediction)
    db.add(new_prediction)
    db.commit()
    db.refresh(new_prediction)
    return request


"""@app.post("/predict/form")
def form_post(request: Request,
              variance: float = Form(...),
              skewness: float = Form(...),
              curtosis: float = Form(...),
              entropy:  float = Form(...),
              db: Session = Depends(get_db)):

    result = Notes(variance=variance,skewness=skewness,curtosis=curtosis,entropy=entropy)
    db.add(result)
    db.commit()
    db.refresh(result)
    return templates.TemplateResponse('form.html', context={'request': request, 'result': result})"""





@app.delete('/{note_id}')
def delete_note(note_id:int, note: BankNote, db: Session = Depends(get_db)):

    note_model=db.query(Models.Notes).filter(Models.Notes.Id==note_id).first()

    if note_model is None:
        raise HTTPException(
            status_code=404,
            detail=f"ID {note_id} : Does Not Exist"
        )

    db.query(Models.Notes).filter(Models.Notes.Id == note_id).delete()
    db.commit()


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
"""@app.post('/predict')
def predict_banknote(note: BankNote, db: Session = Depends(get_db)):

    note_model = Models.Notes()
    note_model.variance = note.variance
    note_model.skewness = note.skewness
    note_model.curtosis = note.curtosis
    note_model.entropy = note.entropy

    data = note.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']


    # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])


    print(prediction)
    print(prediction[0])
    if (prediction[0] > 0.5):
        prediction = "Fake note"
        note_model.note_class = 1
        note_model.prediction= "Fake note"

    else:
        prediction = "Its a Bank note"
        note_model.prediction = "Its a Bank note"
        note_model.note_class = 0

    db.add(note_model)
    db.commit()

    return {
        'prediction': prediction
    }"""
def predict_note_authentication(variance, skewness, curtosis, entropy):
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values

    """

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)
    return prediction


def main(note:BankNote,db: Session = Depends(get_db)):
    st.title("Bank Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    note_model = Models.Notes()
    note_model.variance = note.variance
    note_model.skewness = note.skewness
    note_model.curtosis = note.curtosis
    note_model.entropy = note.entropy

    st.markdown(html_temp, unsafe_allow_html=True)
    note_model.variance = st.text_input("Variance", "Type Here")
    note_model.skewness = st.text_input("skewness", "Type Here")
    note_model.curtosis = st.text_input("curtosis", "Type Here")
    note_model.entropy = st.text_input("entropy", "Type Here")
    result = ""
    if st.button("Predict"):
        note_model.note_class = predict_note_authentication(variance, skewness, curtosis, entropy)[0]
    st.success('The output is {}'.format(note_model.note_class[0]))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
    if (note_model.note_class[0] > 0.5):
        note_model.prediction = "Fake note"
        note_model.note_class = 1


    else:
        note_model.prediction = "Its a Bank note"
        note_model.note_class = 0

    db.add(note_model)
    db.commit()




# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000


# uvicorn app:app --reload