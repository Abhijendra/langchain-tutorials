from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

llm = HuggingFaceEndpoint(repo_id='openai/gpt-oss-120b', task='text-generation')

model = ChatHuggingFace(llm=llm)

# SCHEMA
class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description='Return sentiment of the feedback either Positive or Negative') 

parser = PydanticOutputParser(pydantic_object=Feedback)

str_parser = StrOutputParser()

prompt = PromptTemplate(
            template='Classify the sentiment from the following text into Positive or Negative \n{text} \n{format_instruction}',
            input_variables=['text'],
            partial_variables={'format_instruction': parser.get_format_instructions()})

classifier_chain = prompt | model | parser 

# result = classifier_chain.invoke({'text':'product is not up to mark but usable. Price is cheap so its okay.'})
# print(result.sentiment)

prompt2 = PromptTemplate(template='Write an appropriate response for this positive feedback \n {feedback}',
                         input_variables=['feedback'])

prompt3 = PromptTemplate(template='Write an appropriate response for this negative feedback \n {feedback}',
                         input_variables=['feedback'])

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | str_parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | str_parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'text':'''
This latest version of Echo Dot may be a better version than previous ones, but I am thorough disappointed with Alexa Echo Dot. Many times it does not understand the command, many times it simply says Sorry I don't understand, many times while music is on it is unable to catch the voice hence you have to shout to activate the volume sensors for a different command. Majority of the music is picked up only from Amazon, no generic search for the song, you give a song name and artist name, many times it says that it's unable to find the song.
I just had a quick glimpse of ChatGPT. The level of AI used is immaculate...it actually mind boggling.

I bought ALEXA recently with a perception that ALEXA may have similar level of inbuilt AI, but let's say, ALEXA is far far away from being closest to AI.

In first few minutes, I got to know and I declared it as a complete DUMBO.

I repent. May be I was overexpecting. ALEXA may have good features for home automation, etc, however if you really want to use it majorly for your music needs, I would suggest you better drop the idea of going for ALEXA.

I wonder how a person like Rajiv Makhni gives a complete Thumbs Up for this product and unlike earlier times, provides biased reviews without naming the device drawbacks

Hope above details help some friends to make the right decision.
'''}))
