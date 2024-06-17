import os
import openai
from openai import OpenAI
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
search_tool = SerperDevTool()

class Agent:
    def __init__(self, name, role, backstory, goal, tools=None):
        self.name = name
        self.backstory = backstory
        self.goal = goal
        self.role = role
        self.memory = []
        self.tools = tools if tools else []
        self.message_box = []
    # adding memory for the agent to store recent tasks and outputs 
    def add_to_memory(self, entry):
        self.memory.append(entry)

    # sending messages to other agents
    def send_message(self, recipient, message):
        recipient.message_box.append((self.name, message))
    
    # reading the messages sent from other agents before performing task
    # this is done by removing messages from message box and added to memory
    def read_messages(self):
        while self.message_box:
            sender, message = self.message_box.pop(0)
            self.add_to_memory(f"message from the {sender}: {message}")

    # we now define the function that will do the task assigned
    # reading messages and adding task to the memory first
    # the agent will take up the specialised role assigned and querry gpt3.5 

    def do_task(self, task, inputs):
        self.read_messages()         
        task_info = task.info 
        self.add_to_memory(f"doing task: {task_info}")
         
        '''for the research agent, the search_tool will be assigned to the agent
            which it will be able to use to do a google search online'''

        if 'search_tool' in self.tools:
            search_query = task_info
            search_results = search_tool.run(query=search_query)
            inputs['search_results'] = search_results
            task_info += f"\n\nsearch results:\n{search_results}"

        llm_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"you are a {self.role}. {self.backstory} Your goal is {self.goal}."},
                {"role": "user", "content": task_info}
            ]
        )
        output = llm_response.choices[0].message.content
        self.add_to_memory(f"task output: {output}")

        return output

