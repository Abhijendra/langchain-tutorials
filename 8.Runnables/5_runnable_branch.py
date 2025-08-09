from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableBranch
load_dotenv()

llm = HuggingFaceEndpoint(repo_id='google/gemma-2-2b-it', task='text-generation')

model = ChatHuggingFace(llm=llm)

prompt_1 = PromptTemplate(template='Write a report about the topic: {topic}',
               input_variables=['topic'])

prompt_2 = PromptTemplate(template='Summarize the report:\n {text}',
               input_variables=['text'])

parser = StrOutputParser()

report_gen_chain = prompt_1 | model | parser 

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 50, prompt_2 | model | parser),
    RunnablePassthrough()
)

chain = report_gen_chain | branch_chain

result = chain.invoke({'topic':'India vs USA'})

print(result)