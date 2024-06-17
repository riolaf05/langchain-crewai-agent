from crewai import Agent, Task, Crew, Process

#define agents
data_collector = Agent(
    name="researcher",
    role="Climate Data Collector",
    goal="Collect comprehensive climate data from multiple sources.",
    backstory="You gather climate data on temperature, carbon dioxide levels and other variables relevant to climate change, from reliable sources.",
    tools=['search_tool']
)

data_analyst = Agent(
    name="Data Scientist",
    role="Climate Data Scientist",
    goal="Analyse the collected climate data to identify significant trends.",
    backstory="You analyse climate data to find significant trends and understand the impact of various factors on climate change.",
    tools=[]
)

report_writer = Agent(
    name="Report Writer",
    role="Senior Scientific Report Writer",
    goal="Generate a comprehensive report on climate change findings.",
    backstory="You write detailed scientific reports based on the analysed climate data, highlighting key findings and implications.",
    tools=[]
)

peer_reviewer = Agent(
    name="Peer Reviewer",
    role="Scientific Peer Reviewer",
    goal="Review the scientific report for accuracy, clarity, and completeness.",
    backstory="You review scientific reports to ensure they are accurate, clear, and meet the standards for scientific publication.",
    tools=[]
)

final_report_writer = Agent(
    name="Final Report Writer",
    role="Final Report Writer",
    goal="Incorporate peer review feedback and finalize the scientific report.",
    backstory="You finalize the scientific report by incorporating feedback from peer reviewer and ensure it is publication ready.",
    tools=[]
)

#create tasks
collect_data = Task(
  description=(
    "Identify the all the data you can in {topic}."
    "Focus on identifying big trend in temperature, CO2 levels, and precipitation and causes."
    "Your final report should clearly articulate the key points,"
    "its market opportunities, and potential risks."
  ),
  expected_output='A comprehensive 3 paragraphs long report on the latest AI trends.',
  tools=['search_tool'],
  agent=data_collector,
)


analyse_data = Task(
    info=(
        "Using the following climate data, analyze for trends and patterns:\n{previous_result}\n"
        "1. Identify significant trends in temperature, CO2 levels, and precipitation.\n"
        "2. Determine potential causes of observed trends.\n"
        "3. Summarize key findings in a detailed analysis report."
    ),
    expected_output="Detailed analysis report on climate data trends and potential causes.",
    agent=data_analyst,
    name="Data Analysis"
)

write_report = Task(
    info=(
        "Using the following analysis report, write a comprehensive scientific report on climate change findings:\n{previous_result}\n"
        "1. Include an introduction, methodology, results, discussion, and conclusion.\n"
        "2. Use clear and precise language suitable for a scientific audience.\n"
        "3. Ensure all findings are supported by data and analysis."
    ),
    expected_output="Comprehensive scientific report on climate change findings.",
    agent=report_writer,
    name="Report Writing"
)

review_report = Task(
    info=(
        "Using the following scientific report, review for accuracy, clarity, and completeness:\n{previous_result}\n"
        "1. Ensure the report adheres to scientific standards.\n"
        "2. Check for any errors or inaccuracies in data and analysis.\n"
        "3. Provide feedback and suggestions for improvement."
    ),
    expected_output="Reviewed and revised scientific report, ready for publication.",
    agent=peer_reviewer,
    name="Peer Review"
)

finalize_report = Task(
    info=(
        "Using the following peer-reviewed report, incorporate feedback and finalize the scientific report:\n{previous_result}\n"
        "1. Address all feedback and suggestions provided by the peer reviewer.\n"
        "2. Ensure the report is polished and ready for publication.\n"
        "3. Provide the final version of the scientific report."
    ),
    expected_output="Finalized scientific report, ready for publication.",
    agent=final_report_writer,
    name="Finalize Report"
)

#create the chatbot
ClimateResearchSystem = TheArchitect(
    agents=[data_collector, data_analyst, report_writer, peer_reviewer, final_report_writer],
    tasks=[collect_data, analyse_data, write_report, review_report, finalize_report]
)

result = ClimateResearchSystem.process(inputs={
    "topic": "Climate Change",
    "search": "latest climate data trends"
})

print(result)