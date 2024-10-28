from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

MODEL_NAME="llama3-70b-8192"
TEMPERATURE=0


# llm = ChatGroq(
#    temperature=TEMPERATURE, 
#    model_name=MODEL_NAME
#    )
llm = ChatOpenAI(temperature=0.0,
                 model="gpt-3.5-turbo",
                 max_tokens=512
    )