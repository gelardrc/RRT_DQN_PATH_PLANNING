import rospy
import roslaunch
import random
import os
import time
import numpy as np
from numpy.linalg import norm
from math import *
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from std_msgs.msg import Bool,String
from geometry_msgs.msg import PoseStamped,Point,Quaternion,TwistStamped,Pose,Vector3,Vector3Stamped
from sensor_msgs.msg import LaserScan,PointCloud2

from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

from gazebo_msgs.srv import SetModelState,GetModelState,GetWorldProperties
from gazebo_msgs.msg import ModelState,ModelStates


class Network(nn.Module):

    def __init__(self,input_size,nb_action):
        super(Network,self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size,100) ## MELHOR 100_1_
        self.fc2 = nn.Linear(100,nb_action)
        #self.fc3 = nn.Linear(212,29)
        #self.fc4 = nn.Linear(29,nb_action)
        #self.fc4 = nn.Linear(30,nb_action)
    
    def foward(self,state):
        x = F.relu(self.fc1(state))
        #y = F.relu(self.fc2(x))
        #z = F.relu(self.fc3(y))
        q_values = self.fc2(x)
        return q_values

class ReplayMemory(object):

    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self,event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    def sample(self,batch_size):
        samples = zip(*random.sample(self.memory,batch_size))
        #print('samoples',list(samples))
        return map(lambda x:Variable(torch.cat(x,0)),samples)

class Dqn(object):

    def __init__(self,input_size,nb_action,gamma,start,goal,limites):

        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size,nb_action)
        self.memory = ReplayMemory(capacity = 100000)
        self.optimizer = optim.Adam(params = self.model.parameters(),lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.last_distance = 9999999999999
        self.start = start
        self.goal = goal
        self.limites = limites
        self.td_loss_array = []
    def select_action(self,state):
        var_state = Variable(state, volatile = True)
        self.s = self.model.foward(var_state)*100
        probs = F.softmax(self.s)# t = 100
        action = probs.multinomial(len(probs))
        return action.data[0,0]

    def learn(self,batch_states,batch_actions,batch_rewards,batch_next_states):
        batch_outputs = self.model.foward(batch_states).gather(1,batch_actions.unsqueeze(1)).squeeze(1)
        batch_next_outputs = self.model.foward(batch_next_states).detach().max(1)[0]
        batch_targets = batch_rewards + self.gamma*batch_next_outputs
        td_loss = F.smooth_l1_loss(batch_outputs,batch_targets)
        #self.td_loss_array.append(td_loss)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        self.optimizer.step()
    
    def update(self,state,new_reward):
        #print('state',state)
        new_state = torch.Tensor(state).float().unsqueeze(0)   ## transforma state em um tensor
        #print('new_state',new_state)
        self.memory.push((self.last_state,torch.LongTensor([int(self.last_action)]),torch.Tensor([self.last_reward]),new_state)) ## adiciona a memoria
        new_action = self.select_action(new_state) ## seleciona a saida
        if len(self.memory.memory) > 100:
            batch_states,batch_actions,batch_rewards,batch_next_states = self.memory.sample(100)
            self.learn(batch_states,batch_actions,batch_rewards,batch_next_states)
            #print('fui treinada --> fudeu')
        self.last_state = new_state
        self.last_action = new_action
        self.last_reward = new_reward
        self.reward_window.append(new_reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return new_action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)
    
    def save(self):
        
        torch.save({'state_dict':self.model.state_dict(),
                     'optimizer':self.optimizer.state_dict},
                   'last_brain.pth')
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint ...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("loaded brain")
        else:
            print('no brain file found ...')
    
    def get_reward2(self,state,obj_lista):
        
        angulo   = state[0]
        sensores = state[1:6]        
        
        #if 
        
        
        
        return self.reward
    
    def get_reward(self,state,obj_lista):
        
        state = state[0:3]
        
        if obj_lista.count(state) > 0 :
            
            self.reward = -1 ## colidiu com objeto
        
        elif norm(np.array(state)-np.array(self.goal))==0 :
            
            self.reward = 1 ## chegou no alvo 
            
        elif state[0] > self.limites[0] or state[0] < 0 or state[1] > self.limites[1] or state[1] < 0 or state[2] > self.limites[2] or state[2] < 0 : 
            
            self.reward = -1 ## saiu do enviroment
            
        #elif norm(np.array(state) - np.array(self.goal)) < self.last_distance :
        #   
        #   self.reward = -1/(norm(np.array(state) - np.array(self.mundo.target)))
        #    
        #   self.last_distance = norm(np.array(state) - np.array(self.goal))
       
        #else:
        #   
        #   self.reward = -0.1
       #
        else:
            
            self.reward = -(norm(np.array(state) - np.array(self.goal)))
        

        return self.reward
        
    def define_state(self,pose,sensores):
        
        t = np.concatenate((np.array(pose),sensores),axis=None)
        
        return t
    
    def define_state_angle(self,p,sensores):
        
        
        if norm(p-np.array([0,0,0])) == 0:
            p = np.array([0.001,0.001,0.001])
            ab = p[0]*self.goal[0] + p[1]*self.goal[1] + p[2]*self.goal[2]
            m_a = sqrt(p[0]**2 + p[1]**2 + p[2]**2)
            m_b = sqrt(self.goal[0]**2 + self.goal[1]**2 + self.goal[2]**2)

            #print('ab',ab)

            #print('ma,mb',m_a,m_b)
        
            angle = np.arccos(round((ab)/(m_a*m_b),4))

            t = np.concatenate((angle,sensores),axis=None)

        else:    
            ab = p[0]*self.goal[0] + p[1]*self.goal[1] + p[2]*self.goal[2]
            m_a = sqrt(p[0]**2 + p[1]**2 + p[2]**2)
            m_b = sqrt(self.goal[0]**2 + self.goal[1]**2 + self.goal[2]**2)

            #print('ab',ab)

            #print('ma,mb',m_a,m_b)
        
            angle = np.arccos(round((ab)/(m_a*m_b),4))
        
            t = np.concatenate((angle,sensores),axis=None)
        
        return t

class Drone(object):

    def __init__(self,limites,objs,start,goal):
       self.ros_topics()
       ## inicializa os sensores ##
       self.sensores = Sensors(limites = limites)
       ## Inicializa os subscribers do vant ##
       self.action = Action()

       self.missao = Mission()
       
       self.brain = Dqn(input_size = 7,nb_action= 6,gamma = 0.99,start = start,goal= goal,limites = limites)

       self.x,self.y,self.z = [0,0,0]

       self.pose = [0,0,0]

       pass
    
    def get_pose_cb(self,state):
        self.x,self.y,self.z = [round(state.pose.position.x),round(state.pose.position.y),round(state.pose.position.z)] 
        self.pose = np.array([self.x,self.y,self.z],dtype= int)
        #print('pose',self.x,self.y,self.z)
        pass
    
    def ros_topics(self):
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped,self.get_pose_cb)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        pass

    def reset(self):
        self.action.arm_cmd.value = False
        self.action.arming_client.call(self.action.arm_cmd)
        
        cmd_vel = TwistStamped()
        cmd_vel.header.stamp = rospy.Time()
        cmd_vel.header.frame_id = 'world'
        cmd_vel.twist.linear = Vector3(0,0,0)
        cmd_vel.twist.angular = Vector3(0,0,0)

        self.action.velocidade.publish(cmd_vel)#reset velocity
        
        p = Point(0,0,0) ## trocar despois para initial pose ou novo_aleatorio
        
        o = Quaternion(0,0,0,0) ## manter
        
        acc = Vector3Stamped()
        acc.header.stamp = rospy.Time()
        acc.header.frame_id = 'world'
        acc.vector = Vector3(0,0,0)


        self.action.acelarecao.publish(acc)


        self.action.offb_set_mode.custom_mode = 'AUTO_RTL'

        if(self.action.set_mode_client.call(self.action.offb_set_mode).mode_sent == True):
                rospy.loginfo("AUTO_RTL enabled")
        
        
        self.set_state(ModelState(
                    model_name = 'iris_rplidar',
		            pose = Pose(position = p ,orientation = o)
                    )) #reset pose
        pass

class Sensors(object):
    def __init__(self,limites):
        self.lidar = Lidar()
        self.fakelidar = Fakelidar(limites = limites)
        self.camera = Camera()
        pass
    
class Lidar(object):
    def __init__(self):
        rospy.Subscriber('/agent/lidar', LaserScan,self.get_readings_cb)
    def get_readings_cb(self,readings):
        self.readings = readings.ranges        

class Fakelidar(object):
    
    def __init__(self,limites): ## n_lidar = numero de lidars
        self.limites = limites
        
    def get_readings(self,pose,objs): ## d_pose = drone pose
        
        self.readings = [0,0,0,0,0,0]

        dire = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
        
        
        for index,value in enumerate(dire):
        
            aux = np.add(pose,value)
            
            ## fora do ambiente sensor le a parede == 1 ##
            
            if aux[0] > self.limites[0] or  aux[0] < 0 or aux[1] > self.limites[1] or aux[1] < 0 or aux[2] > self.limites[2] or aux[2] < 0 : 
                
               #self.readings[0][index] = 1
               self.readings[index] = 1 
            
            
            ## se estiver ocupado por um objeto gaha 1 ##
            ## e preciso usar o to list pq o np.add transforma em array ## 
            if aux.tolist() in objs:
              
              self.readings[index] = 1 
              #self.readings[0][index] = 1
                
        return self.readings
        
class Camera(object):
    def __init__(self):
        rospy.Subscriber('/agent/camera/depth/points', PointCloud2,self.get_depth_cb)
        pass
    def get_depth_cb(self,readings):
        self.depth = readings
        #print(self.depth)
        pass

class Action(object):
    
    def __init__(self):
        
        self.current_state = State()
        self.p = PoseStamped()

        self.last_req = rospy.Time.now()
        
        self.offb_set_mode = SetModeRequest()
        self.offb_set_mode.custom_mode = 'OFFBOARD'

        self.arm_cmd = CommandBoolRequest()
        self.arm_cmd.value = True
        

        self.local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
    
        self.state_sub = rospy.Subscriber("mavros/state", State, callback = self.get_state_cb)

        rospy.wait_for_service("/mavros/cmd/arming")
        
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)    

        
        rospy.wait_for_service("/mavros/set_mode")
        
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

        self.velocidade = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel',TwistStamped,queue_size=10)
        
        self.acelarecao = rospy.Publisher('/mavros/setpoint_accel/accel',Vector3Stamped,queue_size=10)
        

        self.rate = rospy.Rate(20)

        # Send a few setpoints before starting
        for i in range(100):   
            if(rospy.is_shutdown()):
                break
            self.local_pos_pub.publish(self.p)
            self.rate.sleep()
        
        pass
    
    def get_state_cb(self,state):
        self.current_state = state
        pass
    
    def envia_acao(self,target):
        #print('t',(rospy.Time.now() - self.last_req))
        #print('armou',self.current_state.armed)
        self.arming_client(CommandBoolRequest(value = True))
        #print('mode',self.current_state.mode)
        if(self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req) > rospy.Duration(5.0)):
            if(self.set_mode_client.call(self.offb_set_mode).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")
            self.last_req = rospy.Time.now()
        else:
            if(not self.current_state.armed and (rospy.Time.now() - self.last_req) > rospy.Duration(5.0)):
                self.arming_client(CommandBoolRequest(value = True))
                if(self.arming_client.call(self.arm_cmd).success == True):
                    rospy.loginfo("Vehicle armed")
                self.last_req = rospy.Time.now()

        self.p.header.stamp = rospy.Time.now()
        self.p.header.frame_id = "agent"
        self.p.pose.position = Point(target[0],target[1],target[2])
        self.p.pose.orientation = Quaternion(0,0,0,0)
        
        self.local_pos_pub.publish(self.p)

class Mission(object):
    def __init__(self):
        self.receiver = rospy.Subscriber("/Mission_planner/rrt_path", String, self.callback)
        self.path = []
        self.old_data = "qualquer_coisa"
        pass
    def callback(self,data):
        ## isso e uma gambiarra pra funcionar com uma mesagem do tipo string ##
        if self.old_data != data.data: # esse if e so pra garantir que nao vai ter repeticao de dados 
            new_array = data.data.split(',')
            for i in range(len(new_array)):
              if (i+1)%3 == 0:
                self.path.append([int(new_array[i-2]),int(new_array[i-1]),int(new_array[i])])
            self.old_data = data.data

class Objs(object):
    
    def __init__(self,limites):
        
        self.field = np.zeros((limites[0],limites[1],limites[2]),dtype = int)

        self.obj_svr = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties) 

        self.get_objs()

        self.att_field()

    pass
    
    def get_objs(self):

        obj_lista = self.obj_svr()

        self.obj_lista = obj_lista.model_names

    def att_field(self):
        self.z_occ = []
        
        for obj in self.obj_lista:
            t = Model(name = obj)
            for i in t.z_occ:
                self.z_occ.append(i)
        pass

class Model(object):
    def __init__(self,name):
        
        self.name = name

        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        modelo = self.get_state(self.name,"world")

        self.center_pose = modelo.pose.position
        
        self.occ()

        pass

    def occ(self):
        
        self.z_occ = []
        
        if self.name[0:3] == 'oak':
            #size = [10,8,6]
            size = [5,4,3]
        
        elif self.name[0:3] == 'pic':
            #size = [8,3,2]
            size = [4,2,1]
        
        else:
            size = [0,0,0]


        for x in range(int(self.center_pose.x)-size[0],int(self.center_pose.x)+size[0]):
            for y in range(int(self.center_pose.y)-size[1],int(self.center_pose.y)+size[1]):
                for z in range(int(self.center_pose.z)-size[2],int(self.center_pose.z)+size[2]):
                    self.z_occ.append([x,y,z])
        pass

class World(object):
    
    def __init__(self,limites,start,goal):
        self.px4 = rospy.Publisher('/agent/restart_px4', Bool, queue_size=10)
        self.px4_ready = rospy.Subscriber('/world_builder/reset_environment', Bool,self.ready_cb)
        self.start = start
        self.goal  = goal
        #self.z_occ = Objetos(n_o = 0,start = start,stop = goal,limites = limites) 
        self.objs = Objs(limites = limites)
        self.drone = Drone(limites = limites,objs = self.objs.z_occ,start = start,goal = goal)

    def ready_cb(self,data):
        self.ready = data.data    

    def reset_world(self):
        
        print('restart')

    def restart(self):
        data = String()
        data.data = 'restart'
        self.px4.publish(data)
    
    def proseguir(self):
        
        pass

def train():

    start = [2,2,2]

    m = World(limites = [10,10,10],start = start,goal = [5,5,5])

    dire = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]

    #start = [2,4,0]

    if m.ready == True:
        print('estou_pronto')

    ### takeoff ##
    while norm(np.array([2,2,2])-np.array(m.drone.pose))!=0:
                m.drone.action.envia_acao(target = [2,2,2])
                time.sleep(2)

    m.goal = m.drone.pose
    ####
    while (not rospy.is_shutdown()):
        events = 2
        for i in range(events):
            steps = 2
            print(norm(np.array(start)-np.array(m.drone.pose)))
            ### forcar posicao inicial ##
            while norm(np.array(start)-np.array(m.drone.pose))!=0:
                m.drone.action.envia_acao(target = start)
                time.sleep(2)
            
            while m.ready == False:
                print('Aguardando_reiniciar')
            for i in range(steps):
                state = m.drone.brain.define_state_angle(m.drone.pose,m.drone.sensores.fakelidar.get_readings(pose = m.drone.pose,objs = m.objs.z_occ))
                new_action = m.drone.brain.update(state = state,new_reward = m.drone.brain.get_reward(state = state.tolist(),obj_lista = m.objs.z_occ))
                
                if m.drone.brain.reward == -1: 
                    ## reinicia gazebo 
                    m.px4.publish(Bool(data=True))
                    break
                
                if m.drone.brain.reward == 1:
                    break  
                
    #while(not rospy.is_shutdown()):
    #    
    #    
    #    m.drone.brain.last_distance = 9999999999999
    #    #while norm(m.drone.pose-np.array(m.goal)):
    #        ## construct state
    #    state = m.drone.brain.define_state_angle(start,m.drone.sensores.fakelidar.get_readings(pose = start,objs = []))
    #    
    #    new_action = m.drone.brain.update(state = state,new_reward = m.drone.brain.get_reward(state = state.tolist(),obj_lista = []))
#
    #    #print(new_action)
#
    #    start[0] = start[0] + dire[new_action][0]
    #    start[1] = start[1] + dire[new_action][1]
    #    start[2] = start[2] + dire[new_action][2]
#
    #    while start[2] < 2:
    #        new_action = m.drone.brain.update(state = state,new_reward = m.drone.brain.get_reward(state = state.tolist(),obj_lista = []))
    #        start[0] = start[0] + dire[new_action][0]
    #        start[1] = start[1] + dire[new_action][1]
    #        start[2] = start[2] + dire[new_action][2]
#
    #    if norm(np.array(m.drone.pose)-np.array(m.goal)) == 0:
    #        #print('cheguei',start)
    #        #print('score',m.drone.brain.score())
    #        #path.append(pathy)
    #        #m.drone.brain.save()
    #        break
    #    print('start',start)
    #    print('actual_pose',m.drone.pose)
    #    while norm(np.array(start)-m.drone.pose)!=0:
    #            m.drone.action.envia_acao(target = start)
#
        #if world.init == True:
        #    world.drone.action.envia_acao(target = [10,11,12])
        #    for i in world.objs : 
        #        if norm(np.array([i[0],i[1],i[1]])-np.array([world.drone.x,world.drone.y,world.drone.z]))<1:
        #            world.reset_world()
        
        #world.objects.att_field() ## atualiza o field 
        
    #    #for i in drone.missao.path:
    #    #    while norm(np.array([i[0],i[1],i[1]])-np.array([drone.x,drone.y,drone.z]))>0.1:
    #    #        drone.action.envia_acao(target = [i[0],i[1],i[1]])
    #    #
    #    #drone_missao_old = drone.missao.path
    rospy.spin()

    pass
 
def teste():
    
    m = World(limites = [10,10,10],start = [2,2,2],goal = [5,5,5])
    dire = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
    m.drone.brain.load()
    #print('ss',m.drone.sensores.fakelidar.get_readings(pose = m.drone.pose,objs = m.objs.z_occ))
    
    while norm(np.array([2,2,2])-np.array(m.drone.pose))!=0:
                print('p',norm(np.array([2,2,2])-np.array(m.drone.pose)))
                print('drone p ',m.drone.pose)
                m.drone.action.envia_acao(target = [2,2,2])
                #time.sleep(5)

    
    while (not rospy.is_shutdown()):
        #print('quase la',norm(np.array(m.drone.pose)-np.array(m.goal)))
        while norm(np.array(m.drone.pose)-np.array(m.goal)) > 0.5:
            
            state = m.drone.brain.define_state_angle(p = m.drone.pose,sensores = m.drone.sensores.fakelidar.get_readings(pose = m.drone.pose,objs = m.objs.z_occ))
            print('estado',state)   
            action = m.drone.brain.select_action(torch.Tensor(state).float().unsqueeze(0))
            print('acao',dire[action])

            next = np.add(dire[action],m.drone.pose)
            print('next',next.tolist())
            while norm(next-np.array(m.drone.pose))!=0:
                m.drone.action.envia_acao(target = next.tolist())
            
            print('p',m.drone.pose)






if __name__ == '__main__':

    print("Inicializando Agente....")

    
    rospy.init_node('Agent', anonymous=False)   
    
    teste()
    #train()  
	
