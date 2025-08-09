from langchain_community.document_loaders import TextLoader

loader = TextLoader('requirements.txt')

doc = loader.load()

print(doc)
