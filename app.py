# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI, Depends,HTTPException
from BankNotes import BankNote
import numpy as np
import pickle
import Models
import pandas as pd
from Database import engine, SessionLocal
from sqlalchemy.orm import Session

Models.Base.metadata.create_all(bind=engine)


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
def index(db: Session = Depends(get_db)):
    return db.query(Models.Notes).all()

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
""""@app.post('/')
def get_note(note: BankNote, db: Session = Depends(get_db)):

    note_model= Models.Notes()
    note_model.variance= note.variance
    note_model.skewness= note.skewness
    note_model.curtosis= note.curtosis
    note_model.entropy= note.entropy

    db.add(note_model)
    db.commit()

    return note"""


""""@app.put('/{note_id}')
def update_note(note_id:int, note: BankNote, db: Session = Depends(get_db)):

    note_model=db.query(Models.Notes).filter(Models.Notes.Id==note_id).first()

    if note_model is None:
        raise HTTPException(
            status_code=404,
            detail=f"ID {note_id} : Does Not Exist"
        )

    note_model.variance = note.variance
    note_model.skewness = note.skewness
    note_model.curtosis = note.curtosis
    note_model.entropy = note.entropy

    db.add(note_model)
    db.commit()

    return note"""

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
@app.post('/predict')
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
    if (prediction[0] > 0.5):
        prediction = "Fake note"
        note_model.prediction = "Fake Note"
    else:
        prediction = "Its a Bank note"
        note_model.prediction = "Its a Bank note"

    db.add(note_model)
    db.commit()

    return {
        'prediction': prediction
    }




# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=2000)

# uvicorn app:app --reload
