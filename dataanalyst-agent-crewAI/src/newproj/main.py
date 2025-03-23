#!/usr/bin/env python
import sys
import os
import pandas as pd
from io import StringIO

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

print(sys.path)

from newproj.crew import QuestCrew, EDACrew,JuniorDACrew


os.environ["OPENAI_API_KEY"] = ""






def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def generate_markdown_for_images(directory):
    # Get list of PNG files in the directory
    images = [img for img in os.listdir(directory) if img.endswith('.png')]

    # Start the markdown section
    markdown_content = "### Plots \n\n"

    # Loop through each image and generate markdown
    for img in images:
        image_path = os.path.join(directory, img)
        markdown_content += f"![{img}]({image_path})\n\n"

    return markdown_content

def run():
    """
    Run the crew.
    """

    print(f"Current working directory: {os.getcwd()}")

    # Initilize the agentops for tracing of communication with LLMs
    #agentops.init(auto_start_session=False)

    # Read the YAML config file
    #with open('config.yaml', 'r') as f:
    #    config = yaml.safe_load(f)    # Read the config file

    # Reading configuarations
    #metadata_txt = read_file(config['app']['Metadata'])
    #datapath = config['app']['DataPath']
    #imagepath = config['app']['ImagePath']
    #num_questions = config['app']['NumOfQuestions']

    metadata_txt = read_file("chd-metadata.txt")
    datapath = "./SAheart_data.csv"
    imagepath = "./mount_point/"
    cleandata_location = "./mount_point/SAheart_clean_data.csv"
    num_questions = 2
    

    print(f"Metadata file location: {metadata_txt}")
    print(f"Datapath location: {datapath}")
    print(f"Plots location: {imagepath}")   

    dclean_inputs = {
        'datapath_info': datapath,
        'cleandata_path': cleandata_location,
    }
    #Run the agent
    crew_output = JuniorDACrew().crew().kickoff(inputs=dclean_inputs)




    # To store analysis for each questions.
    md_content = []
    
    #agentops.start_session( tags = ['question', 'hypothesis'] )

    # Creating hypothesis or generating questions using QuestCrew
    q_inputs = {
        'how_many': int(num_questions),
        'metadata_info': metadata_txt,
    }

    # Run the agent
    qresult = QuestCrew().crew().kickoff(inputs=q_inputs)

    #agentops.end_session("Success")

    if qresult is not None:        
        print(f"Raw result from crew.kickoff: {qresult.raw}")
    
    qlist = qresult.pydantic
    for q in qlist.questions:
        print(f"Question: {q}")            

    # Create the directories for storing the created plots
    for i in range(len(qlist.questions)):
        os.makedirs(f"{imagepath}/q_{i}", exist_ok=True)   

    # Creating the EDACrews for generate code and answer those questions
    # Also to summarize the questions
    
    eda_inputs_list = [{
        'question_str': q,
        'metadata_info': metadata_txt,
        'datapath_info': datapath,
        'imagepath_dir': f"{imagepath}/q_{i}"} for i, q in enumerate(qlist.questions)]

    #agentops.start_session(tags = ['answer', 'analysis'])

    # Run the agent
    final_results = EDACrew().crew().kickoff_for_each(inputs = eda_inputs_list)

    #agentops.end_session("Success")

    # Consolidating the writing to a file. 
    try:
        with open("./mount_point/final_analysis.md", 'w') as file:
            file.write("# Exploratory Data Analysis" + '\n\n')
            file.write("## Dataset Description" + '\n\n')
            file.write(metadata_txt + '\n\n')
            for i, mdc in enumerate(final_results):
                file.write(f"## EDA Analysis - {i+1}" + '\n\n')
                print(mdc.raw)
                file.write(mdc.raw + '\n\n')
                file.write(generate_markdown_for_images(f"{imagepath}/q_{i}") +"\n\n")
        print(f"Successfully wrote to final_analysis.md")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

    # agentops.end_session("Success")



if __name__ == "__main__":
    run()


