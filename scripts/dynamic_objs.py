#!/usr/bin/env python3

import rospy
import numpy as np
import time
import sys
import random


from gazebo_msgs.srv import SpawnModel,DeleteModel,SetModelState,GetModelState
from geometry_msgs.msg import Pose,Point,Quaternion,Wrench,Vector3
from gazebo_msgs.msg import ModelState

#

## Fazer um spwan de algum obj no gazebo usando aqui ##
class obj:
	def __init__(self,pose,name,map,caminho):
		self.mapa = map
		self.sentido = random.randint(0,50)/100
		self.caminho = caminho
		self.velocidade_x = 0
		self.initial_pose = pose
		self.pose = Pose(position= Point(self.initial_pose[0],self.initial_pose[1],self.initial_pose[2]),orientation=Quaternion(0,0,0,0))
		self.name = name
		self.spawm()
	
	def spawm(self):
		spawn_model_client(
	    	model_name=self.name,
	    	model_xml=open(self.caminho, 'r').read(),
	    	robot_namespace='/foo',
	    	initial_pose=self.pose,
	    	reference_frame='world'
		)

		pass
	def atualiza(self):	 
		
		estado = get_state(self.name,'world')
		## isso quer dizer que estou dentro do mapa ##
		if estado.pose.position.x >= self.mapa.x or estado.pose.position.x <= -self.mapa.x:
			
			self.sentido = -self.sentido 
		new_pose = [estado.pose.position.x+self.sentido,estado.pose.position.y,estado.pose.position.z]
		
		set_state(
			ModelState(
				model_name = self.name,
				pose = Pose(position = Point(new_pose[0],new_pose[1],new_pose[2]) ,orientation = Quaternion(0,0,0,0))
				))
		
class mapa:

	def __init__(self,size):

		self.x,self.y,self.z = size


def main():
	map = mapa([5,5,5])

	
	c1 = obj(pose =[random.randint(-map.x,map.x),random.randint(-map.y,map.y),random.randint(0,map.z)],name="bolota",map=map,caminho= cubo_path)

	c2 = obj(pose =[random.randint(-map.x,map.x),random.randint(-map.y,map.y),random.randint(0,map.z)],name="bolota2",map=map,caminho = cubo_path)

	c3 = obj(pose = [random.randint(-map.x,map.x),random.randint(-map.y,map.y),random.randint(0,map.z)],name="bolota3",map=map,caminho = cubo_path)
	
	c4 = obj(pose = [random.randint(-map.x,map.x),random.randint(-map.y,map.y),random.randint(0,map.z)],name="bolota4",map=map,caminho = cubo_path)

	while True:
		c1.atualiza()
		c2.atualiza()
		c3.atualiza()
		c4.atualiza()

if __name__ == '__main__':

	print("Inicializando....")
	
	rospy.init_node('dynamic_objs', anonymous=True)

	rospy.Rate(10)
	
	rospy.wait_for_service('/gazebo/spawn_sdf_model')
	rospy.wait_for_service('/gazebo/get_model_state')
	rospy.wait_for_service('/gazebo/set_model_state')



	spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
	get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
	set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)


	
	delete_model_client = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)


	cubo_path  = '/home/gelo/model_editor_models/simple_box/model.sdf'

	main()
	