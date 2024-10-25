from crewai import Crew, Process
from crewai_kit.agents import financial_researcher, financial_writer, blog_researcher, blog_writer
from crewai_kit.tasks import research_task, write_task, financial_task1, financial_task2

# Forming the tech-focused crew with some enhanced configurations
blog_crew = Crew(
  agents=[blog_researcher, blog_writer],
  tasks=[research_task, write_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True
)

financial_crew = Crew(
    agents=[financial_researcher, financial_writer],
    tasks=[financial_task1, financial_task2],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
)

# rag_crew = Crew(
#     agents=[Router_Agent, Retriever_Agent, Grader_agent, hallucination_grader, answer_grader],
#     tasks=[router_task, retriever_task, grader_task, hallucination_task, answer_task],
#     verbose=True,

# )