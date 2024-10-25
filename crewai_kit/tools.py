from crewai_tools import BaseTool, tool
# import boto3
import os
import json
from crewai_tools import PDFSearchTool, WebsiteSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from utils.database_managers import QDrantDBManager
from utils.embedding import EmbeddingFunction

MODEL_NAME="Llama3-8b-8192"
TEMPERATURE=0
SEARCH_RESULTS=5
GROQ_API_KEY=os.getenv('GROQ_API_KEY')
QDRANT_URL=os.getenv('QDRANT_URL')
QDRANT_API_KEY=os.getenv('QDRANT_API_KEY')
COLLECTION_NAME=os.getenv('COLLECTION_NAME')

# lambda_client = boto3.client('lambda', region_name=os.getenv('AWS_REGION'))
function_name_1 = 'riassume-turnon-ec2-lambda'
function_name_2 = 'riassume-turnoff-ec2-lambda'
payload = {}
llm = ChatGroq(
   temperature=TEMPERATURE, 
   model_name=MODEL_NAME
   )
# llm = ChatOpenAI(temperature=0.0,
#                  model="gpt-3.5-turbo",
#                  max_tokens=512
#     )


embedding = EmbeddingFunction('openAI').embedder
qdrantClient = QDrantDBManager(
    url=QDRANT_URL,
    port=6333,
    collection_name=COLLECTION_NAME,
    vector_size=1536,
    embedding=EmbeddingFunction('openAI').embedder,
    record_manager_url="sqlite:///record_manager_cache.sql"
)
vectore_store_client=qdrantClient.vector_store.as_retriever()

# rag_tool = PDFSearchTool(pdf='attenstion_is_all_you_need.pdf',
#     config=dict(
#         llm=dict(
#             provider="groq", # or google, openai, anthropic, llama2, ...
#             config=dict(
#                 model=MODEL_NAME,
#                 # temperature=0.5,
#                 # top_p=1,
#                 # stream=true,
#             ),
#         ),
#         embedder=dict(
#             provider="huggingface", # or openai, ollama, ...
#             config=dict(
#                 model="BAAI/bge-small-en-v1.5",
#                 #task_type="retrieval_document",
#                 # title="Embeddings",
#             ),
#         ),
#     )
# )

## CUSTOM TOOLS CLASSES

# class QdrantRetrievalTool(BaseTool):
#     name: str = "Qdrant Retrieval Tool"
#     description: str = "Qdrant Retrieval Tool"

#     def _run(self, query: str) -> str:
#         return vectore_store_client.run(query)
#         return vectore_store_client.run(query)['result'] #FIXME


# class TurnOnEC2Tool(BaseTool):
#     name: str = "Turn on EC2"
#     description: str = "This tool turn on EC2 machine on the AWS Cloud."

#     def _run(input : str) -> str:
#         # Convert the payload to JSON
#         payload_json = json.dumps(payload)

#         # Invoke the Lambda function
#         response = lambda_client.invoke(
#             FunctionName=function_name_1,
#             InvocationType='RequestResponse',  # Use 'Event' for asynchronous invocation
#             Payload=payload_json
#         )
#         # Read the response
#         response_payload = response['Payload'].read()
#         response_dict = json.loads(response_payload)
#         print(response_dict)
#         return "Started!"

# TOOLS

web_search_tool = TavilySearchResults(k=SEARCH_RESULTS)
website_rag_tool = WebsiteSearchTool() #https://docs.crewai.com/tools/websitesearchtool
# vision_tool = VisionTool() #https://docs.crewai.com/tools/visiontool
# retrieval_rag_tool = QdrantRetrievalTool()

@tool("retrieval_rag_tool")
def retrieval_rag_tool(query: str) -> str:
    """This tools return informations about Infosys ."""
    # Function logic here
    return vectore_store_client.run(query)


@tool
def router_tool(question):
  """Router Function"""
  if 'self-attention' in question:
    return 'vectorstore'
  else:
    return 'web_search'
