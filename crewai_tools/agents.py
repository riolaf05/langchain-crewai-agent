from crewai import Agent
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv(override=True)

MODEL_NAME="Llama3-8b-8192"
TEMPERATURE=0

search_tool = SerperDevTool()

llm = ChatGroq(temperature=TEMPERATURE, model_name=MODEL_NAME)

#define agents
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
  tools=[search_tool],
  allow_delegation=True
)

# Creating a writer agent with custom tools and delegation capability
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
  tools=[search_tool],
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
  tools=[search_tool],
  allow_delegation=False
)

financial_researcher = Agent(
    role="Lead Financial Analyst",
    goal="Uncover insights about different tech companies",
    backstory="""You work at an asset management firm.
  Your goal is to understand tech stocks like Infosys.""",
    verbose=True,
    allow_delegation=False,
    tools=[query_tool],
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
        "If the answer gnerated is not relevant then perform a websearch using 'web_search_tool'"
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)