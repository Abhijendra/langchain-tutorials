from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([
    ('system','You are a helpful {domain} expert'),
    ('human','Define the topic {topic} in simple terms')
])

prompt = chat_template.invoke({'domain': 'cricket', 'topic':'Dusra'})

print(prompt)
