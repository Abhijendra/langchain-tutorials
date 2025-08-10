from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./9.DocLoaders/rent-agreement.pdf')

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0
)

result = splitter.split_documents(docs)

print(type(result))
print(len(result))
print(result[0].page_content)