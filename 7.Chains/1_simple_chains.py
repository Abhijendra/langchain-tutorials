from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='google/gemma-2-2b-it', task='text-generation')

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(template='Write a 10 lines essay on {topic} for a {age} years old',
               input_variables=['topic', 'age'])


parser = StrOutputParser()

chain = template | model | parser 

query_dict = {'topic':'cow', 'age':5}
result = chain.invoke(query_dict)

print(result)
print('----------------------------')
print(template.invoke(query_dict))

# print the chain
chain.get_graph().print_ascii()