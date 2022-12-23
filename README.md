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


___________________________________________________________________________________________________________________________________________________


**1 - Requirements :**  <a name="requeriments"></a>

Firsts things first, you need to have install into you computer (offboard controller) ROS melodic. Just follow the ROS Wiki tutorials above: 

http://wiki.ros.org/melodic/Installation/Ubuntu

http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment

With your ROS system installed and catkin space configured, we will install PX4 ros package. Follow the tutorials in PX4 site above : 

(I recommend to install via ubuntu_sim_ros_melodic.sh, and to download Firmware into your catkin space !!)

https://dev.px4.io/v1.11_noredirect/en/setup/dev_env_linux_ubuntu.html

Now install mavros (in this work I installed from source, without problem)






