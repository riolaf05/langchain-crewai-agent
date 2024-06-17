from crewai_tools import BaseTool
import boto3
import os
import json

lambda_client = boto3.client('lambda', region_name=os.getenv('AWS_REGION'))
function_name_1 = 'riassume-turnon-ec2-lambda'
function_name_2 = 'riassume-turnoff-ec2-lambda'
payload = {}

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