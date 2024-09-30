# from crewai import Agent, Crew, Process, Task
# from crewai.project import CrewBase, agent, crew, task


from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, llm, tool 
from crewai_tools import CodeInterpreterTool
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL 
from pydantic import BaseModel, Field
from typing import List

from newproj.tools.custom_tool import PythonREPLTool

# Uncomment the following line to use an example of a custom tool
# from newproj.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool



class QuestionList(BaseModel):
  questions: List[str]


pyrepltool = PythonREPL()

python_repl_tool = PythonREPLTool()


@CrewBase
class QuestCrew():
  """Quest crew"""
  agents_config = 'config/quest/agents.yaml'
  tasks_config = 'config/quest/tasks.yaml'

  @llm
  def llm_model(self):
    return ChatOpenAI(temperature=0.0,  # Set to 0 for deterministic output
                      model="gpt-4o-mini",  # Using the GPT-4 Turbo model
                      max_tokens=8000) 
 
  @agent
  def business_consultant(self) -> Agent:
    return Agent(
      config=self.agents_config['business_consultant'],
      max_rpm=None,
      verbose=True
    )

  @task
  def generate_questions_task(self) -> Task:
    return Task(
      config=self.tasks_config['generate_questions_task'],
      output_pydantic = QuestionList
    )

  @crew
  def crew(self) -> Crew:
    """Creates the Llmeda crew"""
    question_crew = Crew(
      agents=self.agents,
      tasks=self.tasks, # Automatically created by the @task decorator
      process=Process.sequential,
      verbose=True,
      output_log_file = "qgen.log"
      # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
    )
  
    return question_crew



@CrewBase
class EDACrew():
  """EDA crew"""
  agents_config = 'config/eda/agents.yaml'
  tasks_config = 'config/eda/tasks.yaml'

  @llm
  def llm_model(self):
    return ChatOpenAI(temperature=0.0,  # Set to 0 for deterministic output
                    model="gpt-4o-mini",  # Using the GPT-4 Turbo model
                    max_tokens=8000) 
 

 
  # return ChatGroq(
  #  model="llama3.1-70b-versatile",
  #  temperature=0.0,
  #  max_retries=2,
  # ) 

  @agent
  def data_scientist(self) -> Agent:
    return Agent(
      config=self.agents_config['data_scientist'],
      verbose=True
    )
 
  @task
  def datascience_task(self) -> Task:
    return Task(
      config=self.tasks_config['datascience_task'],
      tools=[python_repl_tool]
     
    )
 
  @crew
  def crew(self) -> Crew:
    """Creates the eda crew"""
    eda_crew = Crew(
      agents=self.agents,
      tasks=self.tasks, # Automatically created by the @task decorator
      tools=[python_repl_tool],
      process=Process.sequential,
      verbose=True,
      output_log_file = "eda.log"
      # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
    )

    return eda_crew