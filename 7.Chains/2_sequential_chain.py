from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='google/gemma-2-2b-it', task='text-generation')

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(template='Write an essay on {topic} for a {age} years old',
               input_variables=['topic', 'age'])

prompt2 = PromptTemplate(template='Give most important 5 lines from the {paragraph}',
                         input_variables=['paragraph'])

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser 

query_dict = {'topic':'Football', 'age':10}
result = chain.invoke(query_dict)

print(result)

