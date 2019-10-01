# Reinforcement learning to solve the Halite challenge

Halite is a programming game played by two or four players, which compete for the resources of the map. To win the game one must have more resources than his opponents at the end of the game. The game ends after a certain number of turns, that can vary from match to match, but usually is about 400 or 500 turns.

Introduction and initial committment: Reinforcement_learning_for_halite.pdf

**Achievements:** 

- implemented the halite environment engine in python, compatible with the openAI gym standard;
- partially solved the resource collection task for a single ship and a single player with tabular methods; 
- implemented the multi-agent framework for tabular methods.

# Run project in a conda virtual environment

> conda create -n \<env_name> python=3.7.4

Now choose the folder in which you want to clone the github repository

> git clone https://github.com/nicoladainese96/haliteRL.git
>
> cd haliteRL
>
> conda activate \<env_name>
>
> pip install -r requirements.txt
>
> python
>
> \>\>\> import sys; sys.executable

Copy the path \<path> without quotes (e.g. /home/nicola/anaconda3/envs/test-env/bin/python)

> \>\>\>  quit()

>sudo \<path> -m ipykernel install --name \<env_name>
>
>jupyter notebook

Open the notebooks that you want to inspect and select \<env_name> kernel.

# Run project with Docker

> docker run --rm -it -p 8888:8888 nicoladainese96/halite_rl:v0.0 bash
>
> jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
>
> open the last URL in a browser and work in jupyter notebook

exit to close the docker

**Explanation first command:**  

**--rm**    removes the container once you exit from it (the instance is deleted, just the image remains) 

**-it**   opens an interactive session with the container  

**-p <host_port>:<container_port>**   exposes a port for host-container communication (8888 is the default one used for jupyter)  

**\<user>/\<repo>:\<tag>**   image to be run (see https://hub.docker.com/r/nicoladainese96/halite_rl/tags for all updated tags)  
  
**bash**   type of shell used in the interactive session  

Feel free to change the optional arguments of this command (e.g. you can pull the image and then run it locally; also you can work inside it and commit the changes if you remove the --rm keyword).
