from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

# PROMPT 1 : DETAILED REPORT
prompt_template_1 = PromptTemplate(
    template='Generate a detailed report on the topic {topic}',
    input_variables=['topic']
)

# PROMPT 2 : 5 LINES SUMMARY REPORT
prompt_template_2 = PromptTemplate(template='Give 5 lines summary for the text: {text}', 
                                   input_variables=['text'])

parser = StrOutputParser()

chain = prompt_template_1 | model | parser | prompt_template_2 | model | parser 

result = chain.invoke({'topic':'G20 Summit'})

print(result)