from crewai_tools.crews import crew

# Starting the task execution process with enhanced feedback
result = crew.kickoff(inputs={'topic': 'GenAI in Digital Twin'})
print(result)