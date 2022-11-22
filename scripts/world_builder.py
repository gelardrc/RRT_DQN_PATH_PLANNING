import rospy
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

def cb_reset(data):
    if data.data==True:
        print('calbacj')
        p = Point(-1,8,0)
        o = Quaternion(0,0,0,0)
        set_state(ModelState(
        model_name = "iris_rplidar",
        pose = Pose(position = p ,orientation = o)
        ))
    

def spawn(model_name,model,pose = [random.randint(-10,10),random.randint(-10,10),random.randint(0,10)]):
    p = Point(pose[0],pose[1],pose[2])
    o = Quaternion(0,0,0,0)
    pose = Pose(position= p,orientation=o)
    spawn_model_client(
	    	model_name=model_name,
	    	model_xml=open(model, 'r').read(),
	    	robot_namespace='/foo',
	    	initial_pose= pose,
	    	reference_frame='world'
		)

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

def constroi_mundo():
    n_arvores = 20
    count_x = 0
    count_y = 0
    
    delete_model_client('ground_plane')

    p = Point(-1,8,0)
    o = Quaternion(0,0,0,0)
    set_state(ModelState(
    model_name = "iris_rplidar",
	pose = Pose(position = p ,orientation = o)
    ))

    spawn(model_name = 'grass_plane',model=grass,pose=[0,0,0])

    delete_model_client('asphalt_plane')
    
    for i in range(n_arvores):
        spawn(model_name = 'oak_tree'+str(i),model = oak_tree,pose = [0+count_x,0+count_y,0])
        count_x += 10
        if  count_x == 100:
            count_y += 16
            count_x = 0 

def main():
    
    print("No inicializado....")
    
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

    if constroi : constroi_mundo()
    
    while True:
        ## Atualiza as posicoes dos carrinhos ## 
        for car in trucks:
            car.atualiza(velocidade = random.random())
        
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                print('Reiniciando_poses')
                reset(Empty())
                main()
        except:
            continue
       

    rospy.spin()
    
    pass

if __name__ == '__main__':

    print("Inicializando....")

    rospy.init_node('world_builder', anonymous=False)   
    
    rospy.wait_for_service('/gazebo/get_model_state')
    
    rospy.wait_for_service('/gazebo/set_model_state')

    get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    delete_model_client = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

    spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    
    reset = rospy.ServiceProxy('/gazebo/reset_world',Empty)

    constroi = True

    iris = "/home/gelo/src/Firmware/Tools/sitl_gazebo/models/iris/iris.sdf"

    bebop = "/home/gelo/.gazebo/models/parrot_bebop_2/model.sdf"

    pickup = "/home/gelo/.gazebo/models/pickup/model.sdf"

    oak_tree = "/home/gelo/.gazebo/models/oak_tree/model.sdf"

    grass = "/home/gelo/.gazebo/models/grass_plane/model.sdf"

    house = ""

    rospy.Subscriber('/gazebo/reset_uav_pose',Bool,cb_reset)

    main()  
	
