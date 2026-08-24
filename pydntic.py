from pydantic import BaseModel

class Spider(BaseModel):
    name: str='Peter Parker'
    age: int

peter={'name':'Tom Holland', 'age': 25}

x=Spider(**peter)

print(x)