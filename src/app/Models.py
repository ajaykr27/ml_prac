from sqlalchemy import Column,Integer, Float, String
from Database import Base

class Notes(Base):
    __tablename__= "banknote1"

    Id = Column(Integer,primary_key=True,index=True)
    variance = Column(Float)
    skewness = Column(Float)
    curtosis = Column(Float)
    entropy = Column(Float)
    note_class = Column(Integer)
    prediction = Column(String(40))
