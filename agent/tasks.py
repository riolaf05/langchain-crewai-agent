from crewai import Task
from agent.agents import researcher, writer
from agent.tools import TurnOnEC2
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
load_dotenv(override=True)

search_tool = SerperDevTool()
turn_on_ec2_tool = TurnOnEC2()

# Research task
research_task = Task(
  description=(
    "Identify the next big trend in {topic}."
    "Focus on identifying pros and cons and the overall narrative."
    "Your final report should clearly articulate the key points,"
    "its market opportunities, and potential risks."
  ),
  expected_output='A comprehensive 3 paragraphs long report on the latest AI trends.',
  tools=[search_tool],
  agent=researcher,
)

# Writing task with language model configuration
write_task = Task(
  description=(
    "Compose an insightful article on {topic}."
    "Focus on the latest trends and how it's impacting the industry."
    "This article should be easy to understand, engaging, and positive."
  ),
  expected_output='A 4 paragraph article on {topic} advancements formatted as markdown.',
  tools=[search_tool],
  agent=writer,
  async_execution=False,
  output_file='new-blog-post.md'  # Example of output customization
)

technician_task = Task(
  description=(
    "Turn on EC2 instances"
    "EC2 are AWS VMs that can be turned of programmatically"
    "Use the provided tool to turn on the EC2 VM and send confirmation."
  ),
  expected_output='A confirmation message or an error message.',
  tools=[turn_on_ec2_tool],
  agent=writer,
  async_execution=False,
)