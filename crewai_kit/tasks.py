from crewai import Task
from crewai_kit.agents import blog_writer, blog_researcher, financial_researcher, financial_writer, stock_data_retriever, stock_analyst, stock_reporter
from crewai_kit.tools import web_search_tool
# from crewai_kit.tools import TurnOnEC2, SerperDevTool
from dotenv import load_dotenv
load_dotenv(override=True)

# Research task
research_task = Task(
  description=(
    "Identify the next big trend in {topic}."
    "Focus on identifying pros and cons and the overall narrative."
    "Your final report should clearly articulate the key points,"
    "its market opportunities, and potential risks."
  ),
  expected_output='A comprehensive 3 paragraphs long report on the latest AI trends.',
  tools=[web_search_tool],
  agent=blog_researcher,
)

# Writing task with language model configuration
write_task = Task(
  description=(
    "Compose an insightful article on {topic}."
    "Focus on the latest trends and how it's impacting the industry."
    "This article should be easy to understand, engaging, and positive."
  ),
  expected_output='A 4 paragraph article on {topic} advancements formatted as markdown.',
  tools=[web_search_tool],
  agent=blog_writer,
  async_execution=False,
  output_file='new-blog-post.md'  # Example of output customization
)

# technician_task = Task(
#   description=(
#     "Turn on EC2 instances"
#     "EC2 are AWS VMs that can be turned of programmatically"
#     "Use the provided tool to turn on the EC2 VM and send confirmation."
#   ),
#   expected_output='A confirmation message or an error message.',
#   tools=[turn_on_ec2_tool],
#   agent=blog_writer,
#   async_execution=False,
# )

# router_task = Task(
#     description=("Analyse the keywords in the question {question}"
#     "Based on the keywords decide whether it is eligible for a vectorstore search or a web search."
#     "Return a single word 'vectorstore' if it is eligible for vectorstore search."
#     "Return a single word 'websearch' if it is eligible for web search."
#     "Do not provide any other premable or explaination."
#     ),
#     expected_output=("Give a binary choice 'websearch' or 'vectorstore' based on the question"
#     "Do not provide any other premable or explaination."),
#     agent=Router_Agent,
#     tools=[router_tool],
# )

# retriever_task = Task(
#     description=("Based on the response from the router task extract information for the question {question} with the help of the respective tool."
#     "Use the web_serach_tool to retrieve information from the web in case the router task output is 'websearch'."
#     "Use the rag_tool to retrieve information from the vectorstore in case the router task output is 'vectorstore'."
#     ),
#     expected_output=("You should analyse the output of the 'router_task'"
#     "If the response is 'websearch' then use the web_web_search_tool to retrieve information from the web."
#     "If the response is 'vectorstore' then use the rag_tool to retrieve information from the vectorstore."
#     "Return a claer and consise text as response."),
#     agent=Retriever_Agent,
#     context=[router_task],
#    #tools=[retriever_tool],
# )

# grader_task = Task(
#     description=("Based on the response from the retriever task for the quetion {question} evaluate whether the retrieved content is relevant to the question."
#     ),
#     expected_output=("Binary score 'yes' or 'no' score to indicate whether the document is relevant to the question"
#     "You must answer 'yes' if the response from the 'retriever_task' is in alignment with the question asked."
#     "You must answer 'no' if the response from the 'retriever_task' is not in alignment with the question asked."
#     "Do not provide any preamble or explanations except for 'yes' or 'no'."),
#     agent=Grader_agent,
#     context=[retriever_task],
# )

# hallucination_task = Task(
#     description=("Based on the response from the grader task for the quetion {question} evaluate whether the answer is grounded in / supported by a set of facts."),
#     expected_output=("Binary score 'yes' or 'no' score to indicate whether the answer is sync with the question asked"
#     "Respond 'yes' if the answer is in useful and contains fact about the question asked."
#     "Respond 'no' if the answer is not useful and does not contains fact about the question asked."
#     "Do not provide any preamble or explanations except for 'yes' or 'no'."),
#     agent=hallucination_grader,
#     context=[grader_task],
# )

# 

financial_task1 = Task(
    description="""Conduct a comprehensive analysis of Infosys's use of artificial intelligence in their integrated annual report 2022-23.""",
    expected_output="Full analysis report in bullet points",
    agent=financial_researcher,
)

financial_task2 = Task(
    description="""Using the insights provided, develop an engaging blog
  post that highlights the importance of artificial answer_task = Task(
#     description=("Based on the response from the hallucination task for the quetion {question} evaluate whether the answer is useful to resolve the question."
#     "If the answer is 'yes' return a clear and concise answer."
#     "If the answer is 'no' then perform a 'websearch' and return the response"),
#     expected_output=("Return a clear and concise response if the response from 'hallucination_task' is 'yes'."
#     "Perform a web search using 'web_web_search_tool' and return ta clear and concise response only if the response from 'hallucination_task' is 'no'."
#     "Otherwise respond as 'Sorry! unable to find a valid response'."),
#     context=[hallucination_task],
#     agent=answer_grader,
#     #tools=[answer_grader_tool],
# )intelligence at Infosys.
  Your post should be informative yet accessible, catering to a casual audience.
  Make it sound cool, avoid complex words.""",
    expected_output="Full blog post of at least 4 paragraphs",
    agent=financial_writer,
)

# Define task function for stock analysis
def analyze_stock(symbol):
  # Data Retriever gathers financial data
  data = stock_data_retriever.run_tool("get_stock_data", symbol=symbol)
  # Analyst analyzes the data (replace with your analysis logic)
  analysis = stock_analyst.ask(f"Analyze trends in {symbol} stock data for the past year", data=data)
  # Reporter generates a textual report
  report = stock_reporter.ask(f"Generate a report summarizing the analysis of {symbol} stock")
  return report
stock_task = Task(description="Analyze a stock", function=analyze_stock, args=["AAPL"])
