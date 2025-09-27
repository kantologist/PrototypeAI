from pydantic import BaseModel
from typing import Optional

class InputData(BaseModel):
    Employed: Optional[bool] = None 
    Bank_Balance: Optional[float] = None
    Annual_Salary: Optional[float] = None 