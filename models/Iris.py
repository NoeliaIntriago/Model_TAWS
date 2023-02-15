from pydantic import BaseModel, conlist
from typing import List

# Modelo para realizar clasificación
class Iris(BaseModel):
    data: List[conlist(float, min_items=4, max_items=4)]