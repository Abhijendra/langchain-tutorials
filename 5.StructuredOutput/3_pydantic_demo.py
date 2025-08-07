from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'Nitish' # default value
    age: Optional[int] = None 
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5.0, description="Decimal value representing CGPA of the student")
    phone: str = Field(max_length=10, min_length=10)

new_student = {'name': 'Abhijendra', 'email':'abc@gmail.cm', 'cgpa':9.77, 'phone': '1234567890'}
# new_student = {'name': 32} # will throw validation error due to pydantic

## setting both values
# new_student = {'name': 'Abhijendra', 'age': 29} 
# new_student = {'name': 'Abhijendra', 'age': '29'} # '29' will automatically typecasted by pydantic to int value 29 -> type coercing 

student = Student(**new_student)

print('Type = ', type(student))
print(student)

# convert to dict
student_dict = dict(student)
print(student_dict['phone'])

# convert to Json
student_json = student.model_dump_json()
print(student_dict['cgpa'])
