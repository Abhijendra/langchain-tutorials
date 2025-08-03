from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

model = ChatOpenAI(model='o3-mini-2025-01-31',
                    # max_tokens=100
                    )
st.header('Research Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# DYNAMIC PROMPT
template = load_prompt('./4.Prompts/template.json')


# STATIC PROMPT
# user_input = st.text_input("Enter your prompt: ")

if st.button('Generate'):
    # st.text('Hello')

    # prompt = template.invoke({
    # 'paper_input': paper_input,
    # 'style_input': style_input,
    # 'length_input': length_input
    # })

    # result = model.invoke(prompt) 

    ## CREATING A CHAIN WITH template and model and invoking it only once

    chain = template | model 

    result = chain.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input        
    })

    st.write(result.content)


