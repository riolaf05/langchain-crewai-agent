from crewai import Crew, Process
from crewai_kit.agents import financial_researcher, financial_writer, blog_researcher, blog_writer
from crewai_kit.agents import markdown_blog_reasarcher, markdown_blog_writer 
# from crewai_kit.agents import stock_data_retriever, stock_analyst, stock_reporter
from crewai_kit.agents import router_agent, retriever_agent, grader_agent, hallucination_grader, answer_grader
from crewai_kit.tasks import research_task, write_task, financial_task1, financial_task2
from crewai_kit.tasks import markdown_blog_task_research, markdown_blog_task_writer
from crewai_kit.tasks import router_task, retriever_task, grader_task, hallucination_task, answer_task

######## Blog 
blog_crew = Crew(
  agents=[blog_researcher, blog_writer],
  tasks=[research_task, write_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True
)

######## Financial analysis
financial_crew = Crew(
    agents=[financial_researcher, financial_writer],
    tasks=[financial_task1, financial_task2],
    memory=True,
    cache=True,
    verbose=True
)

######## 
rag_crew = Crew(
    agents=[router_agent, retriever_agent, grader_agent, hallucination_grader, answer_grader],
    tasks=[router_task, retriever_task, grader_task, hallucination_task, answer_task],
    verbose=True,

)

######## Stock analysis
# stock_crew = Crew(agents=[stock_data_retriever, stock_analyst, stock_reporter], tasks=[stock_task])

######## Markdown blog creation
markdown_blog_crew = Crew(
    agents=[markdown_blog_reasarcher, markdown_blog_writer],
    tasks=[markdown_blog_task_research, markdown_blog_task_writer],
    emory=True,
    cache=True,
    verbose=True
    )