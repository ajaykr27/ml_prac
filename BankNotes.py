# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:51:19 2020

@author: win10
"""
from pydantic import BaseModel,Field
# 2. Class which describes Bank Notes measurements

class BankNote(BaseModel):
    Id= int
    variance= float
    skewness= float
    curtosis= float
    entropy= float
    note_class= int
    prediction= str
