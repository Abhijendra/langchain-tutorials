from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm1 = HuggingFaceEndpoint(repo_id='openai/gpt-oss-120b', task='text-generation')
llm2 = HuggingFaceEndpoint(repo_id='openai/gpt-oss-120b', task='text-generation')

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(template='Generate 5 lines short notes on the provided text \n {text}', input_variables=['text'])
prompt2 = PromptTemplate(template='Generate 5 questions quiz on the provided text \n {text}', input_variables=['text'])
prompt3 = PromptTemplate(template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}', input_variables=['notes', 'quiz'])

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merged_chain = prompt3 | model1 | parser 

chain = parallel_chain | merged_chain

text = '''
    Black holes are among the most mysterious cosmic objects, much studied but not fully understood. These objects aren’t really holes. They’re huge concentrations of matter packed into very tiny spaces. A black hole is so dense that gravity just beneath its surface, the event horizon, is strong enough that nothing – not even light – can escape. The event horizon isn’t a surface like Earth’s or even the Sun’s. It’s a boundary that contains all the matter that makes up the black hole.

    There is much we don’t know about black holes, like what matter looks like inside their event horizons. However, there is a lot that scientists do know about black holes.
    '''

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()