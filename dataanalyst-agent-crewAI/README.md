# Data analyst agent using crewAI

This is a dockerized usage of crewAI for creating a data analysis agent. 
For data analyst code automaton the Original code was used from this blog post: https://medium.com/@manaranjanp/building-a-collaborative-ai-agent-framework-for-automated-eda-using-crewai-351478b424ce
Following Additional steps and modifications were done over the original code:
- Added another crew  in the beginning to print descriptive statistics
- Dockerised the whole project to avoid any crewAI installation issues on mac
- Few miscell changes ( the REPL tool, comment out agentops etc)

### Customizing and building

** you need to change  `OPENAI_API_KEY` required in the src/newproj/main.py (TODO: take openAI key from .env file) **

To build your crewai project do:

```bash

$ docker build -t crewai-dataanalyst .
```

## Running the Project

To kickstart your crew of AI agents and begin task execution, the directory "mountdir" in  current folders  is mapped into the docker container /app/mount_point to write the output



```bash
$  rm -rf ./mountdir/*
$  docker run --mount type=bind,src=./mountdir,dst=/app/mount_point crewai-dataanalyst
```

To run crewAI from within the docker container follow the below steps:
```bash
$  docker run -it   crewai-dataanalyst /bin/bash 
$  cd /app
$  python src/newproj/main.py
```

## Understanding Your Crew

The newproj Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/` and subdirectories underneath has `tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `agents.yaml` file located in subdirectories of `config/`   outlines the capabilities and configurations of each agent in your crew.

## CrewAI Support
You are welcome to fork and  make any changes leveraging crewAI
For  crewAI support.
- Visit  [crewAI documentation](https://docs.crewai.com)
- Reach out to crewAI team through  [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join crewAI Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with crewAI docs](https://chatg.pt/DWjSBZn)

