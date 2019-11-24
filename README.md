# Origin of this project

This is the final project for the exam in [laboratory of computational physics (mod. B)](https://it.didattica.unipd.it/off/2018/LM/SC/SC2443/000ZZ/SCP8082526/N0) course held by professor Marco Baiesi during the master's degree course in [Physics of Data](https://www.unipd.it/en/physics-data). The related presentation can be viewed at this [link](https://docs.google.com/presentation/d/1EJmlQ-L-AC-snDwLroGnENuX4wGkJSlqsndbm1H76gA/edit?usp=sharing). We (me, Stefano Mancone; Nicola Dainese; Francesco Vidaich) really enjoied doing this project, we think that Reinforcement Learning is a wonderful field and we invite anyone that has spare time and it is interested to read the OpenAi SpinningUp RL guide: [link](https://spinningup.openai.com/en/latest/user/introduction.html).

Contacts:
nicola.dainese96@gmail.com
stefanomancone915#gmail.com
francescovidaich@gmail.com

# Reinforcement learning to solve the Halite challenge

Halite is a programming game played by two or four players, which compete for the resources of the map. To win the game one must have more resources than his opponents at the end of the game. The game ends after a certain number of turns, that can vary from match to match, but usually is about 400 or 500 turns.

Introduction and initial committment: Reinforcement_learning_for_halite.pdf

**Achievements:** 

- implemented the halite environment engine in python, compatible with the openAI gym standard;
- partially solved the resource collection task for a single ship and a single player with tabular methods; 
- implemented the multi-agent framework for tabular methods.


**Demo:**

This is a demo of our trained agent in a 7x7 map. The ship has to navigate the sea and collect the halite from the cells of the map. Brighter colors indicate a greater amount of halite and more halite collected when the ship stops over them. At each frame we show the convenience (Q-value) of each move (move in one of the four cardinal directions or stay still and collect resources) with colors ranging from red (less convenient) to green (more convenient). These Q-values of course have been learned autonomously by the agent during training. 

<img src="Tutorials/Support_material/play_episode_HQ.gif">

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
