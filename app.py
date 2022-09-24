# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI, Depends,HTTPException
from BankNotes import BankNote
from starlette.responses import RedirectResponse
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

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

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

Notes=[]

# 3. Index route, opens automatically on http://127.0.0.1:8000

@app.get('/',response_class=HTMLResponse)
async def read_predicts(request: Request, db: Session = Depends(get_db)):
    records = db.query(Models.Notes).all()
    print(records)
    return templates.TemplateResponse("index.html", {"request": request, "data": records})

"""@app.post('/',response_class=HTMLResponse)
def index(request: Request,db: Session = Depends(get_db)):
    response = RedirectResponse('/', status_code=303)
    return response"""

@app.get('/predict',response_class=HTMLResponse)
async def read_predicts(request: Request, db: Session = Depends(get_db)):
    records = db.query(Models.Notes).all()
    print(Models.Notes.prediction)
    return templates.TemplateResponse("base.html", {"request": request, "data": records})


"""@app.get("/predict/{id}", response_class=HTMLResponse)
def read_movie(request: Request, id: BankNote.Id, db: Session = Depends(get_db)):
    item = db.query(Models.Notes).filter(Models.Notes.Id==id).first()
    print(item)
    return templates.TemplateResponse("overview.html", {"request": request, "predict": item})"""

@app.post("/predict/",response_class=HTMLResponse)
async def create_movie(request: Request,db: Session = Depends(get_db), variance: BankNote.variance = Form(...), skewness: BankNote.skewness = Form(...), curtosis: BankNote.curtosis = Form(...), entropy: BankNote.entropy = Form(...)):

    note_class = classifier.predict([[variance, skewness, curtosis, entropy]])
    if (note_class[0] > 0.5):
        prediction = "Its a Fake note"
        note_class = 1
    else:
        prediction = "Its a Bank note"
        note_class = 0


    predictions = Models.Notes(variance=variance, skewness=skewness, curtosis=curtosis, entropy=entropy,note_class=note_class,prediction=prediction)
    db.add(predictions)
    db.commit()
    #response = RedirectResponse('/', status_code=303)
    #return response
    return templates.TemplateResponse("base.html", {"request": request, "prediction":prediction,"variance":variance
                                                    ,"skewness":skewness,"curtosis":curtosis,"entropy":entropy})


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

'''@app.delete('/{note_id}')
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
    }
'''



# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=2000)

# uvicorn app:app --reload