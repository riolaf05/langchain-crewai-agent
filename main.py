from crewai_kit.crews import crew
from dotenv import load_dotenv
import os
load_dotenv(override=True)

# Starting the task execution process with enhanced feedback
result = crew.kickoff(inputs={'topic': 'GenAI in Digital Twin'})
print(result)