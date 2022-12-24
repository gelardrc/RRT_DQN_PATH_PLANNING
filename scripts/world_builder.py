import rospy
import roslaunch
import time
import random
import keyboard
from pcg_gazebo.simulation import create_object
from pcg_gazebo.generators import WorldGenerator
from std_msgs.msg import Bool


from gazebo_msgs.srv import SpawnModel,DeleteModel,SetModelState,GetModelState

from geometry_msgs.msg import Pose,Point,Quaternion

from gazebo_msgs.msg import ModelState

from std_srvs.srv import Empty

import threading


class cmd_px4(object):
    
    def __init__(self):
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/gelo/catkin_python3/src/PX4-Autopilot/launch/gelo.launch"])
        pass
    def start(self):
        print('start')
        self.launch.start()
        rospy.loginfo("started")
    def stop(self):
        self.launch.shutdown()
        rospy.loginfo("shutdown")
    def cmd_start_cb():
        pass

    def cmd_stop_cb():
        pass

class model:
    def __init__(self,name,spawn = False,model= '',pose = [random.randint(-10,10),random.randint(-10,10),random.randint(0,10)]):
        self.name = name
        self.model = model
        
        if spawn:
            self.spawn_model(pose)
        #self.velocidade  = velocidade
        self.t = get_state(self.name,"world")
        self.sentido = True
        self.y_lim_max = 0
        self.y_lim_min = 0
        self.x_lim_min = 0
        self.x_lim_max = 100 
    def spawn_model(self,pose):
        p = Point(pose[0],pose[1],pose[2])
        o = Quaternion(0,0,0,0)
        pose = Pose(position= p,orientation=o)
        spawn_model_client(
	    	model_name=self.name,
	    	model_xml=open(self.model, 'r').read(),
	    	robot_namespace='/foo',
	    	initial_pose= pose,
	    	reference_frame='world'
		)
        pass

    def atualiza(self,velocidade):
        self.velocidade = velocidade
        self.t = get_state(self.name,"world") 
        if self.t.pose.position.x <= self.x_lim_max and self.sentido:
            p = Point(self.t.pose.position.x+ self.velocidade,self.t.pose.position.y,self.t.pose.position.z)
            o = Quaternion(0,0,0,0)
            set_state(ModelState(
            model_name = self.name,
			pose = Pose(position = p ,orientation = o)
            ))
            if self.t.pose.position.x + self.velocidade > self.x_lim_max :
                self.sentido = False

        else: #self.t.pose.position.y >= 12 and not self.sentido:
            p = Point(self.t.pose.position.x - self.velocidade ,self.t.pose.position.y,self.t.pose.position.z)
            o = Quaternion(0,0,0.99,0)
            set_state(ModelState(
            model_name = self.name,
			pose = Pose(position = p ,orientation = o)
            ))
            if self.t.pose.position.x - self.velocidade <= self.x_lim_min :
                self.sentido = True
    
class world(object):
    
    def __init__(self):
        
        self.fcu = cmd_px4()
       
        self.reset = False 
        
        self.init_models()

        self.init_cbs()

        self.sucess.publish(Bool(data=False))
    
    def init_models(self):
        
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.delete_model_client = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        self.spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        self.reset = rospy.ServiceProxy('/gazebo/reset_world',Empty)

        self.iris = "/home/gelo/src/Firmware/Tools/sitl_gazebo/models/iris/iris.sdf"

        self.bebop = "/home/gelo/.gazebo/models/parrot_bebop_2/model.sdf"

        self.pickup = "/home/gelo/.gazebo/models/pickup/model.sdf"

        self.oak_tree = "/home/gelo/.gazebo/models/oak_tree/model.sdf"

        self.grass = "/home/gelo/.gazebo/models/grass_plane/model.sdf"

        self.house = ""
        
        pass
    
    def set_world(self):
        
        self.sucess.publish(Bool(data=False))
        
        print('setting world')
        n_arvores = 20
        count_x = 0
        count_y = 0
    
        self.delete_model_client('ground_plane')

        self.spawn(model_name = 'grass_plane',model=self.grass,pose=[0,0,0])

        self.delete_model_client('asphalt_plane')

        for i in range(n_arvores):
            self.spawn(model_name = 'oak_tree'+str(i),model = self.oak_tree,pose = [0+count_x,0+count_y,0])
            count_x += 10
            if  count_x == 100:
                count_y += 16
                count_x = 0
            pass

        self.sucess.publish(Bool(data=True))
        
        p = Point(2,4,0)
        o = Quaternion(0,0,0,0)
        set_state(ModelState(
        model_name = 'iris_rplidar',
	    pose = Pose(position = p ,orientation = o)
        ))

    def init_cbs(self):
        rospy.Subscriber('/agent/restart_px4',Bool,self.cb_reset)
        self.sucess = rospy.Publisher('/world_builder/reset_environment',Bool,queue_size=10)
        pass
    
    def cb_reset(self,data):
        self.reset = data.data

    def spawn(self,model_name,model,pose = [random.randint(-10,10),random.randint(-10,10),random.randint(0,10)]):
        p = Point(pose[0],pose[1],pose[2])
        o = Quaternion(0,0,0,0)
        pose = Pose(position= p,orientation=o)
        self.spawn_model_client(
	        	model_name=model_name,
	        	model_xml=open(model, 'r').read(),
	        	robot_namespace='/foo',
	        	initial_pose= pose,
	        	reference_frame='world'
	    	)

    def reset_world(self):
        
        self.fcu.stop()
        
        #time.time(10)
        
        self.fcu = cmd_px4()

        self.fcu.start()

        #time.sleep(10)

        self.set_world()

def main():
    
    mundo = world()

    mundo.fcu.start()

    #time.sleep(10)

    mundo.set_world()

    trucks = []
    count_y = 8
    ## Cria os modelos caso esses modelos ja estejam no mapa e so trocar o spawn = False e passar o nome do modelo 
    ## no name e deixar model no default  
    for i in range(1):
        trucks.append(model(name="pickup_"+str(i),
                            spawn = True,
                            model =pickup,
                            pose = [0,count_y,0]))
        count_y += 16 

    
    
    while(not rospy.is_shutdown()):
        #print('mundo reset -->',mundo.reset)
        if mundo.reset == True:
        
            mundo.reset_world()
            
            ## gambiarra do caminhao  melhorar isso um dia
            trucks = []
            count_y = 8
            for i in range(1):
                trucks.append(model(name="pickup_"+str(i),
                            spawn = True,
                            model =pickup,
                            pose = [0,count_y,0]))
                count_y += 16

        for car in trucks:
            
            car.atualiza(velocidade = random.random())
        
        mundo.sucess.publish(Bool(data = True))
    
    pass

if __name__ == '__main__':

    print("Inicializando....")

    rospy.init_node('world_builder', anonymous=False)   
    
    get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    pickup = "/home/gelo/.gazebo/models/pickup/model.sdf"

    main()  
	
