import rospy
import math
import random
import threading


from gazebo_msgs.srv import SpawnModel,DeleteModel,SetModelState,GetModelState

from geometry_msgs.msg import Pose,Point,Quaternion

from gazebo_msgs.msg import ModelState

import threading

class truck:
    def __init__(self,name,velocidade):
        self.name = name
        self.velocidade = velocidade
        self.t = get_state(self.name,"world")
    def percurso(self,waypoints):
            
        ## movimento no y
        while self.t.pose.position.y < waypoints[0]:
            p = Point(self.t.pose.position.x,self.t.pose.position.y + self.velocidade,self.t.pose.position.z)
            o = Quaternion(0,0,0,0)
            set_state(ModelState(
            model_name = self.name,
			pose = Pose(position = p ,orientation = o)
            ))
            # Atualiza 
            self.t = get_state(self.name,"world")
        ## rotacao
        p = Point(self.t.pose.position.x,self.t.pose.position.y,self.t.pose.position.z)
        o = Quaternion(0,0,-0.699,0.71)
        set_state(ModelState(
        model_name = self.name,
		pose = Pose(position = p ,orientation = o)
        ))
        self.t = get_state(self.name,"world")        
        ## atualiza x
        while self.t.pose.position.x <  waypoints[1]:
            p = Point(self.t.pose.position.x+self.velocidade,self.t.pose.position.y,self.t.pose.position.z)
            o = Quaternion(0,0,-0.699,0.71)
            set_state(ModelState(
            model_name = self.name,
			pose = Pose(position = p ,orientation = o)
            ))
            # Atualiza 
            self.t = get_state(self.name,"world")       
        ## rotaciona 
        p = Point(self.t.pose.position.x,self.t.pose.position.y,self.t.pose.position.z)
        o = Quaternion(0,0,0.99,-0.03)
        set_state(ModelState(
        model_name = self.name,
		pose = Pose(position = p ,orientation = o)
        ))
        self.t = get_state(self.name,"world")  
        ## atualiza y  
        while self.t.pose.position.y >  waypoints[2]:
            p = Point(self.t.pose.position.x,self.t.pose.position.y - self.velocidade,self.t.pose.position.z)
            o = Quaternion(0,0,0.99,-0.03)
            set_state(ModelState(
            model_name = self.name,
			pose = Pose(position = p ,orientation = o)
            ))
            # Atualiza 
            self.t = get_state(self.name,"world")
        p = Point(self.t.pose.position.x,self.t.pose.position.y,self.t.pose.position.z)
        o = Quaternion(0,0,0.66,0.74)
        set_state(ModelState(
        model_name = self.name,
		pose = Pose(position = p ,orientation = o)
        ))
        self.t = get_state(self.name,"world")
        while self.t.pose.position.x >  waypoints[3]:
            p = Point(self.t.pose.position.x-self.velocidade,self.t.pose.position.y,self.t.pose.position.z)
            o = Quaternion(0,0,0.66,0.74)
            set_state(ModelState(
            model_name = self.name,
			pose = Pose(position = p ,orientation = o)
            ))
            # Atualiza 
            self.t = get_state(self.name,"world")
        p = Point(self.t.pose.position.x,self.t.pose.position.y,self.t.pose.position.z)
        o = Quaternion(0,0,0,0)
        set_state(ModelState(
        model_name = self.name,
		pose = Pose(position = p ,orientation = o)
        ))
        self.t = get_state(self.name,"world")      
    def atualiza(self):
        self.t = get_state(self.name,"world") 
        if self.t.pose.position.y < -28:
            p = Point(self.t.pose.position.x,self.t.pose.position.y + self.velocidade,self.t.pose.position.z)
            o = Quaternion(0,0,0,0)
            set_state(ModelState(
            model_name = self.name,
			pose = Pose(position = p ,orientation = o)
            ))
            # Atualiza 
        pass
def main():
    #t1 = truck(name= "truck",velocidade = 0.1)
    print("No inicializado....")
    #truck = get_state('truck',"world")
    #print(truck.pose.orientation)
    #t = truck(name="truck",velocidade = random.random())
    t2 = truck(name="truck_clone",velocidade = random.random())
    #t_x = threading.Thread(target=t.percurso(waypoints = [12,7.5,-28,-12,71]), args=(),daemon=True).start()
    #t_2_x = threading.Thread(target=t2.percurso(waypoints = [12,0,-28,36]), args=(),daemon=True).start()
    
    while True:
       t2.percurso(waypoints = [12,0,-28,36]) 

    rospy.spin()
    pass
if __name__ == '__main__':

    print("Inicializando....")

    rospy.init_node('truck_movement', anonymous=False)   
    #rospy.Rate(10)

    rospy.wait_for_service('/gazebo/get_model_state')
    rospy.wait_for_service('/gazebo/set_model_state')   
    
    get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    delete_model_client = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

    main()  
	
