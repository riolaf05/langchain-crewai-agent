from crewai import Crew, Process
from crewai_tools.agents import researcher, writer, technician
from crewai_tools.tasks import research_task, write_task, technician_task

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[researcher, writer],
  tasks=[research_task, write_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True
)