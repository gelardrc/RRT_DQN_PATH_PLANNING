import numpy as np
from numpy.linalg import norm
import random
import rospy
from std_msgs.msg import String

class RRT(object):
    
    def __init__(self,z_start,z_target,dimensions=[],max_int = 900):
        self.space  = Space() # defino o space
        self.missao  = Missao() ## escreve o formato da missao
        self.z_start = z_start # start point
        self.z_target = z_target # goal point
        self.r_tree = Tree(Node(z_start)) # inicializo a arvore com start point
        for t in range(max_int): 
            self.get_z_random() # Pego aleatoriamente um no que pertenca ao espaco 
            self.get_near_node() # defino o no mais perto do gerado aleatorio
            if self.col_free(): # verifico se esse caminho entre eles e livre de colisao
                self.node = Node(self.z_random) ## cria random como um node
                self.r_tree.pertence_arvore(self.node)
                self.node.parent.append(self.z_near) ## adiciona z_near como pai de z_random
                self.z_near.child.append(self.node) ## adiciona z_random como filho de z_near
                self.node.h = self.min_space ## atualiza a distancia entre z_near e z_random
                self.node.g = norm(np.array(self.node.pose)-np.array(self.z_start))
                self.r_tree.add_node(self.node)
        
        self.r_tree.find_minimal_path()
                
        
        pass       

    def col_free(self):
        return True

    def get_near_node(self):
        self.min_space = float('inf')
        distance = []
        for i in self.r_tree.nodes:
            distance.append(norm(np.array(self.z_random) - np.array(i.pose)))
        distance = np.array(distance)
        dmin = min(distance)
        ind_min = distance.tolist().index(dmin)
        self.z_near = self.r_tree.nodes[ind_min] 
        pass

    def get_z_random(self):
        if random.random() <=0.50:
            self.z_random = self.z_target
            #print('prob')
        else:
            self.z_random  = [random.randint(self.space.x_min,self.space.x_max),
                              random.randint(self.space.y_min,self.space.y_max),
                              random.randint(self.space.z_min,self.space.z_max)]

class Node(object):
    def __init__(self,node):
        self.pose = node
        self.x = node[0]
        self.y = node[1]
        self.z = node[2]
        self.parent = []
        self.child = []
        self.h = 0 ## distance between parent and actual node 
        self.g = 0 
        pass

class Tree(object):
    
    def __init__(self,node):
        self.nodes = []
        self.nodes.append(node)
        pass
    
    def add_node(self,node):
        self.nodes.append(node)
        pass
    
    def find_minimal_path(self):
        self.last_node = self.nodes[0]
        actual_node = self.nodes[len(self.nodes)-1]
        self.path = []
        new_node = []
        
        while norm(np.array(self.last_node.pose)-np.array(actual_node.pose)) != 0: ## self.nodes[0] = Z_start
            min_g = float('inf')
            
            for i in actual_node.parent: ##pega todos os itens parents
                if i.g < min_g:  ## verifica qual tem a menor distancia 
                    new_node = i
                    min_g = i.g
            actual_node = new_node
            self.path.append(actual_node.pose)
            self.path.reverse()
        
        pass
    
    def find_paths(self):
        self.first_node = self.nodes[0]
        self.current_node = self.first_node
        path = []
        while norm(np.array(self.current_node) - np.array(self.z_target)) != 0: # enquanto current node != target node
            random.choice()     
               
    def pertence_arvore(self,node):
        if node in self.nodes:
            print('estou na arvore')

class Space(object):
    def __init__(self,dimensions = [[0,10],[0,10],[0,10]]):
        self.x_min = dimensions[0][0]
        self.x_max = dimensions[0][1]
        self.y_min = dimensions[1][0]
        self.y_max = dimensions[1][1]
        self.z_min = dimensions[2][0]
        self.z_max = dimensions[2][1]
        pass

class Missao(object):
    def __init__(self):
        self.missao = rospy.Publisher('/Mission_planner/rrt_path', String, queue_size=10)
        self.path = String()
    def envia_missao(self,targets):
        ## pretendo deixar como str mesmo, no futuro eu mudo
        dumb = ""
        for index,i in enumerate(targets):
          if index == 0:
            dumb += str(i[0])+","+str(i[1])+","+str(i[2])
          elif index == len(targets)-1:
            dumb += ","+str(i[0])+","+str(i[1])+","+str(i[2])
          else:
            dumb += ","+str(i[0])+","+str(i[1])+","+str(i[2])

        self.path.data = dumb
        
        self.missao.publish(self.path)

def rrt_algorithm():
    
    path = RRT(z_start=[0,0,5],z_target=[10,10,10])
    print(path.r_tree.path)
    
    while (not rospy.is_shutdown()):
        path.missao.envia_missao(targets = path.r_tree.path)

    rospy.spin()

    
    pass


if __name__ == '__main__':

    print("Inicializando....mission_planener")

    rospy.init_node('RRT_Mission_planner', anonymous=False) 

    rrt_algorithm()  
	