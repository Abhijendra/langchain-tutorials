from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as  np


load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'Who is god of cricket?'


docs_embedding = embeddings.embed_documents(documents)

query_embedding = embeddings.embed_query(query)

similarity_score_2d = cosine_similarity(X=[query_embedding], Y=docs_embedding)

scores = similarity_score_2d[0]
print(scores)

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(documents[index])
print(f'similarity score is {score}')


