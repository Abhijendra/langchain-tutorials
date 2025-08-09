from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import  RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda
load_dotenv()

llm = HuggingFaceEndpoint(repo_id='google/gemma-2-2b-it', task='text-generation')

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(template='Write a joke about the topic: {topic}',
               input_variables=['topic'])

parser = StrOutputParser()

joke_gen_chain = prompt | model | parser 

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x: len(x.split()))
})

final_chain = joke_gen_chain | parallel_chain

result = final_chain.invoke({'topic':'Linux OS'})

print(result)