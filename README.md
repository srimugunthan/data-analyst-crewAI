Data analyst agent with crewAI
====================================

This is an attempt at using crewAI for building a data analyst agent. 

The steps followed are shown below
![alt text](https://github.com/srimugunthan/data-analyst-crewAI/blob/main/docs/imgcrewaiproject.jpg)

The code is based on this  [Blog post](https://medium.com/@manaranjanp/building-a-collaborative-ai-agent-framework-for-automated-eda-using-crewai-351478b424ce )

Additional changes that were added are:
* Added another crew in the beginning to print descriptive statistics
* Dockerised the whole project to avoid any crewAI installation issues on mac
* Few miscell changes ( the REPL tool, comment out agentops etc)

TODO:
* Add a crew to read from multiple datasources (postgresql, local csv etc)
* Add a crew to do data cleaning  
