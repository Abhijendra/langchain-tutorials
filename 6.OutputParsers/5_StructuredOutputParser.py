from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description='fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(template='Give 3 facts about the {topic} \n{format_instructions}',
                          input_variables=['topic'],
                          partial_variables= {'format_instructions': parser.get_format_instructions()}
                          )

# ===================================== 1st way without using chains =============================

# prompt = template.invoke({'topic':'black hole'})
# result = model.invoke(prompt)
# print(result)
# final_result = parser.parse(result.content)

# ========================================== 2nd way using chains =============================
chain = template | model | parser 

final_result = chain.invoke({'topic':'china vs USA'})

print('--------------------------------------------------')
print(final_result)

