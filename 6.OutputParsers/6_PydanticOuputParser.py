from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Annotated
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='google/gemma-2-2b-it',
                    task='text-generation')

model = ChatHuggingFace(llm=llm)

# SCHEMA
class Person(BaseModel):
    name: str = Field(description='Name of a person')
    age: int = Field(gt=18, description='Age of person')
    city: str = Field(description='A city where person belongs')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(template='Generate the name, age, city of a fictional {nationality} person \n{format_instruction}',
                          input_variables=['nationality'],
                          partial_variables={'format_instruction': parser.get_format_instructions()})

chain = template | model | parser 
query_dict = {'nationality':'sri-lankan'}
result = chain.invoke(query_dict)
print(result)

print('-------------------------------------------------------')
prompt = template.invoke(query_dict)
print(prompt)
# result = model.invoke(prompt)
# print(parser.parse(result.content))