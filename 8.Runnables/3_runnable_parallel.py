from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import  RunnableSequence, RunnablePassthrough, RunnableParallel
load_dotenv()

llm = HuggingFaceEndpoint(repo_id='google/gemma-2-2b-it', task='text-generation')

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(template='Write a joke about the topic: {topic}',
               input_variables=['topic'])

prompt2 = PromptTemplate(template='Explain the joke: {joke}',
               input_variables=['joke'])

parser = StrOutputParser()

joke_gen_chain = prompt1 | model | parser 

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': prompt2 | model | parser
})

final_chain = joke_gen_chain | parallel_chain

result = final_chain.invoke({'topic':'football'})

print(result['joke'])
print()
print(result['explanation'])