from crewai_tools import BaseTool
import boto3
import os
import json
from crewai_tools import PDFSearchTool
from langchain_groq import ChatGroq
from crewai_tools  import tool

lambda_client = boto3.client('lambda', region_name=os.getenv('AWS_REGION'))
function_name_1 = 'riassume-turnon-ec2-lambda'
function_name_2 = 'riassume-turnoff-ec2-lambda'
payload = {}

llm = ChatGroq(temperature=TEMPERATURE, model_name=MODEL_NAME)

@tool
def router_tool(question):
  """Router Function"""
  if 'self-attention' in question:
    return 'vectorstore'
  else:
    return 'web_search'

rag_tool = PDFSearchTool(pdf='attenstion_is_all_you_need.pdf',
    config=dict(
        llm=dict(
            provider="groq", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama3-8b-8192",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="huggingface", # or openai, ollama, ...
            config=dict(
                model="BAAI/bge-small-en-v1.5",
                #task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)

class TurnOnEC2(BaseTool):
    name: str = "Turn on EC2"
    description: str = "This tool turn on EC2 machine on the AWS Cloud."

    def _run(input : str) -> str:
        # Convert the payload to JSON
        payload_json = json.dumps(payload)

        # Invoke the Lambda function
        response = lambda_client.invoke(
            FunctionName=function_name_1,
            InvocationType='RequestResponse',  # Use 'Event' for asynchronous invocation
            Payload=payload_json
        )
        # Read the response
        response_payload = response['Payload'].read()
        response_dict = json.loads(response_payload)
        print(response_dict)
        return "Started!"