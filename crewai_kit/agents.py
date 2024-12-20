from crewai import Agent
from dotenv import load_dotenv
load_dotenv(override=True)
from crewai_kit.tools import retrieval_rag_tool, web_search_tool
from crewai_kit.tools import router_tool
# from crewai_kit.tools import get_stock_data
from crewai_kit.llm import llm

#blog writer
blog_researcher = Agent(
  role='Senior Researcher',
  goal='Uncover groundbreaking technologies in {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "Driven by curiosity, you're at the forefront of"
    "innovation, eager to explore and share knowledge that could change"
    "the world."
  ),
  tools=[web_search_tool],
  allow_delegation=True
)

blog_writer = Agent(
  role='Writer',
  goal='Narrate compelling tech stories about {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "With a flair for simplifying complex topics, you craft"
    "engaging narratives that captivate and educate, bringing new"
    "discoveries to light in an accessible manner."
  ),
  tools=[web_search_tool],
  allow_delegation=False
)

cloud_technician = Agent(
  role='Technician',
  goal='Turn on and off Cloud VMs',
  verbose=True,
  memory=True,
  backstory=(
    "With a knack for optimizing infrastructure, you efficiently manage and automate the lifecycle of virtual machines. Your precision and reliability ensure seamless operations, effortlessly scaling environments up or down to meet the ever-changing demands of dynamic workloads."
  ),
  tools=[web_search_tool],
  allow_delegation=False
)

# Financial analysis
financial_researcher = Agent(
    role="Lead Financial Analyst",
    goal="Uncover insights about different tech companies",
    backstory="""You work at an asset management firm.
  Your goal is to understand tech stocks like Infosys.""",
    verbose=True,
    allow_delegation=False,
    tools=[retrieval_rag_tool],
    llm=llm,
)
financial_writer = Agent(
    role="Tech Content Strategist",
    goal="Craft compelling content on tech advancements",
    backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# markdown blog writer
markdown_blog_reasarcher = Agent(
    role="Computer Science Researcher",
    goal='Understanding the published research paper and extract the main insights',
    backstory="""You work at a Computer Science school.
                  A new paper has been published and you need to extract
                  the main information and provide the main insights to
                  the students.You have a knack for dissecting complex
                  research papers""",
    verbose=True,
    tools=[web_search_tool],
    allow_delegation=False
    )

markdown_blog_writer = Agent(
    role="Computer Science Writer",
    goal= 'Craft easy to understand content for Computer Science students',
    backstory="""You are working on writing a summary of a research paper.
                  Use the content of the Computer Science Researcher to develop
                  a short comprehensive summary of the research paper""",
    verbose=True,
    allow_delegation=False,
)

########## RAG + websearch
router_agent = Agent(
  role='Router',
  goal='Route user question to a vectorstore or web search',
  backstory=(
    "You are an expert at routing a user question to a vectorstore or web search."
    "Use the vectorstore for questions on concept related to Retrieval-Augmented Generation."
    "You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search."
  ),
  verbose=True,
  allow_delegation=False,
  llm=llm,
)

retriever_agent = Agent(
  role="Retriever",
  goal="Use the information retrieved from the vectorstore to answer the question",
  backstory=(
      "You are an assistant for question-answering tasks."
      "Use the information present in the retrieved context to answer the question."
      "You have to provide a clear concise answer."
  ),
  verbose=True,
  allow_delegation=False,
  llm=llm,
)

grader_agent =  Agent(
  role='Answer Grader',
  goal='Filter out erroneous retrievals',
  backstory=(
    "You are a grader assessing relevance of a retrieved document to a user question."
    "If the document contains keywords related to the user question, grade it as relevant."
    "It does not need to be a stringent test.You have to make sure that the answer is relevant to the question."
  ),
  verbose=True,
  allow_delegation=False,
  llm=llm,
)

hallucination_grader = Agent(
    role="Hallucination Grader",
    goal="Filter out hallucination",
    backstory=(
        "You are a hallucination grader assessing whether an answer is grounded in / supported by a set of facts."
        "Make sure you meticulously review the answer and check if the response provided is in alignmnet with the question asked"
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

answer_grader = Agent(
    role="Answer Grader",
    goal="Filter out hallucination from the answer.",
    backstory=(
        "You are a grader assessing whether an answer is useful to resolve a question."
        "Make sure you meticulously review the answer and check if it makes sense for the question asked"
        "If the answer is relevant generate a clear and concise response."
        "If the answer gnerated is not relevant then perform a websearch using 'web_web_search_tool'"
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

########### Stock analysis
# stock_data_retriever = Agent(name="Data Retriever", tools=[get_stock_data])
# stock_analyst = Agent(name="Analyst", llm="text-davinci-003")  # Leverage a powerful LLM
# stock_reporter = Agent(name="Reporter", llm="bard")  # Use a good summarization LLM