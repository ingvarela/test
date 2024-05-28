# -*- coding: utf-8 -*-
"""

@author: Sergio Varela
"""

from pydantic import BaseModel
# 2. Class which describes the features of the flower types
class FlowerClassification(BaseModel):
    SepalLenghtCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float
        
    