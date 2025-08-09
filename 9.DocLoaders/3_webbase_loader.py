from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

url = 'https://economictimes.indiatimes.com/news/international/global-trends/trump-says-us-deploying-nuclear-submarines-in-response-to-provocative-russian-comments/articleshow/123048264.cms'
loader = WebBaseLoader(url)

docs = loader.load()

# print(len(docs))
# print(docs[0].page_content)

model = ChatOpenAI(model='gpt-3.5-turbo-0125')

prompt = PromptTemplate(template='Answer the following question according to the provided text \n {text} \n Who are two dead economies according to Trump?', input_variables=['text'])

parser = StrOutputParser()

chain = prompt | model | parser 

result = chain.invoke({'text': docs[0].page_content})

print(result)