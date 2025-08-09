from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./9.DocLoaders/rent-agreement.pdf')

docs = loader.load()

print(len(docs))

print(docs[0])

