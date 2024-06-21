from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from crewai_tools import SerperDevTool
load_dotenv(override=True)

search_tool = SerperDevTool()

#define agents
researcher = Agent(
  role='Senior Researcher',
  goal='Uncover groundbreaking technologies in {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "Driven by curiosity, you're at the forefront of"
    "innovation, eager to explore and share knowledge that could change"
    "the world."
  ),
  tools=[search_tool],
  allow_delegation=True
)

# Creating a writer agent with custom tools and delegation capability
writer = Agent(
  role='Writer',
  goal='Narrate compelling tech stories about {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "With a flair for simplifying complex topics, you craft"
    "engaging narratives that captivate and educate, bringing new"
    "discoveries to light in an accessible manner."
  ),
  tools=[search_tool],
  allow_delegation=False
)

technician = Agent(
  role='Technician',
  goal='Turn on and off Cloud VMs',
  verbose=True,
  memory=True,
  backstory=(
    "With a knack for optimizing infrastructure, you efficiently manage and automate the lifecycle of virtual machines. Your precision and reliability ensure seamless operations, effortlessly scaling environments up or down to meet the ever-changing demands of dynamic workloads."
  ),
  tools=[search_tool],
  allow_delegation=False
)