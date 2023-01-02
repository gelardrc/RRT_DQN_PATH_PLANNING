# Adaptive Path Planning for UAS Fusing Rapidly-Exploring Random Trees and Deep Reinforcement Learning in an Agriculture Dynamic Environment

**Author:** Gabriel G.R Castro

**Email:** gabriel.guitar@gmail.com

**Abstract** : Unmanned Aerial System (UAS) is a suitable solution for monitoring the growing cultures due to the possibility of covering a large area presented in this kind of scenario and the necessity of periodic monitoring. In inspection and monitoring tasks, the UAS must find an optimal or near optimal collision-free route given initial and target positions. In this sense, path planning strategies are crucial, especially online path planning that can represent the robot’s operational environment or even for controlling purposes. Therefore, this research proposes an online adaptive 3D path planning
solution based on the fusion of Rapidly-Exploring Random Trees (RRT) and Deep Reinforcement Learning (DRL) algorithms to be applied in the autonomous trajectory of UAS during the inspection of an olive-growing environment. The proposed framework was tested in a simulated environment
using Gazebo and ROS. The results showed that the proposed solution accomplished the trial for 91.2% of the test environment.

**Keywords:** Aerial Robots; Multiple Robots; Path Planning; Dynamic Environment; Precision Agriculture.

____________________________________________________________________________________________________________________________________________________

**Index** 

[1.Requeriments](#requeriments)

[2.Project structure](#structure)

[3.Explainning Scripts files](#scripts)

[4.Node structure](#nodes)

[5.Run the project](#run)
___________________________________________________________________________________________________________________________________________________


**1 - Requirements :**  <a name="requeriments"></a>

Firsts things first, you need to have installed into your computer (offboard controller) ROS melodic. Just follow the ROS Wiki tutorials above:

>http://wiki.ros.org/melodic/Installation/Ubuntu
>
>http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment

With your ROS system installed and catkin space configured, we will install PX4 ros package. Follow the tutorials in PX4 site above : 

(I recommend to install via ubuntu_sim_ros_melodic.sh, and to download Firmware into your catkin space !!)

>https://dev.px4.io/v1.11_noredirect/en/setup/dev_env_linux_ubuntu.html

Now install mavros (in this work I installed from source, without problem)

>sudo apt-get install ros-melodic-mavros ros-melodic-mavros-extras

Obs : You will probably need to install other standard ros-packges, like rospy and some gazebo messages. Since they fit in any project, you probably will have them already installed into your ros workspace 

_____________________________________________________________________________________________________________________________________________________

**2 - Project Structure :**  <a name="structure"></a>

This project has the structure of a catkin-ros package where : 

Directorys:

Models &rarr;  Contain the models used in simulations ( most of them are already in gazebo standard models, but these have little changes)

Launch &rarr;  Has the compiled of nodes you need to use to Start simulations and tests for this project

Scripts &rarr;  Is where the actual python and jupyternotebooks are stored

Worlds  &rarr;  Contain the files that describe the worlds in gazebo simulation

data    &rarr;  Contain some information about the data acquired during tests and simulations

______________________________________________________________________________________________________________________________________________________

**3 - Scripts files :**  <a name="scripts"></a>

The work has four major pieces of codes, that can be found in scripts directory. 

![alt text](https://drive.google.com/uc?export=view&id=1uzo2l9fdNdYq-qMNXdIfdAgzsm68Uisf)

**Alg_trieno_python_simu.ipynb &rarr;** Is the notebook used that actually train the DQN, save and create the file last_brain.pth, that is the DQN saved model.

**agent.py &rarr;** Is where we start all things attached to the agent like sensors, mission messages, actions and so on.....

**world_builder &rarr;** Is the core of the simulation, it started gelo.launch (px4+mavros nodes) and create and actualize the environment in gazebo.

**mission_planner.py &rarr;** That is just a normal RRT implementation, but taking into account the models create in world_builder. It also sends a message to agent.py, with the planned mission.

________________________________________________________________________________________________________________________________________________________

**4 - Nodes Structure:**  <a name="nodes"></a>

That is how the nodes communicate to each other using ROS messages and topics.

![alt text](https://drive.google.com/uc?export=view&id=1ja95hUccOdFtAy7Cu1TDQo_8BGi5HkLw)


________________________________________________________________________________________________________________________________________________________

**5 - Run the project**  <a name="run"></a>

> cd your_catkin_workspace
>
> git clone https://github.com/gelardrc/RRT_DQN_PATH_PLANNING.git
>
> copy and paste all /moddels/* to  ~/.gazebo/models  (it's important because world_builder points to ~/.gazebo/models)
>
> copy and paste gelo.launch (maybe you will need to chmod +x gelo.launch ) into PX4-Autopilot/launch
>
> roslaunch px4 gelo.launch  (If you are using px4 package inside catkin, reminder to exports paths, in other to gazebo and ros update paths or just update and source gelo_setup.bash with your own path)

Wait until gazebo opens, and try a takeoff / land commander, to inspect if everything is ok.

> python ./world_builder.py   
>
> python ./mission_planner.py
>
>rosbag record -a
>
> python ./agent.py   

**Obs :** In agent.py you can choose on main function, if you want to test a Dqn or train a Dqn into gazebo world, you just need to call them on main function 




