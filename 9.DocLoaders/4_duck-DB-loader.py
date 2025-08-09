from langchain_community.document_loaders import DuckDBLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = DuckDBLoader("SELECT * FROM read_csv_auto('9.DocLoaders/transactions.csv')")

data = loader.load()

print(len(data))

print(data[0])


