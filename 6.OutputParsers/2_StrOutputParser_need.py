from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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

prompt_1 = prompt_template_1.invoke({'topic':'Black Hole'})
result = model.invoke(prompt_1)

print(result.content)

# PROMPT 2 : 5 LINES SUMMARY REPORT
prompt_template_2 = PromptTemplate(template='Give 5 lines summary for the text: {text}', input_variables=['text'])

prompt_2 = prompt_template_2.invoke({'text':result.content})
result = model.invoke(prompt_2)

print(result.content)