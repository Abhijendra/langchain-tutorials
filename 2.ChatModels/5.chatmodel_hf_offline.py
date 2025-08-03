from langchain_huggingface import   ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    # model_id='meta-llama/Llama-3.2-3B-Instruct', # commented to prevent downloading
    task='text-generation',
    pipeline_kwargs={
        'temperature':0.5,
        'max_new_token':100
    }
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of France?")
print(result.content)