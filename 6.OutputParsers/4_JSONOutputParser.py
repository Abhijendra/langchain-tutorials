from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser =JsonOutputParser()

template = PromptTemplate(template='Give me the name, age and city of a fictional person \n{format_instructions}',
                        input_variables=[],
                        partial_variables={'format_instructions': parser.get_format_instructions()})

# ========================================== 1st way without using chains =============================
# prompt = template.format()
# print(prompt)
# result = model.invoke(prompt)
# print(result)

# final_result = parser.parse(result.content)

# ========================================== 2nd way using chains =============================
chain = template | model | parser 
final_result = chain.invoke({}) # empty dict needed to pass even if nothing in your input_variables

print('---------------------------')
print(type(final_result)) # dict
print(final_result['name'])
