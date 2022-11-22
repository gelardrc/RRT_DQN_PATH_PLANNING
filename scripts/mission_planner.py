import rospy
import time
import dijkstra3d
from  random import random
import numpy as np
from math import *
from numpy.linalg import norm
from pcg_gazebo.simulation import create_object

from gazebo_msgs.srv import GetModelState,GetWorldProperties


class world:
    def __init__(self):
        pass
    def models(self):
        self.models_pose = []
        t = get_models_world()
        t.model_names.pop(0)
        #print(t.model_names)
        for model in t.model_names:
            t = get_state(model,"world")
            self.models_pose.append(t.pose.position)
        
        pass

class Node3D:
    def __init__(self):
        self.p     = [0, 0, 0]
        self.i     = 0
        self.iPrev = 0

def closestNode3D(rrt, p):
    distance = []
    for node in rrt:
        distance.append( sqrt((p[0] - node.p[0])**2 + (p[1] - node.p[1])**2 + (p[2] - node.p[2])**2) )
    distance = np.array(distance)
    
    dmin = min(distance)
    ind_min = distance.tolist().index(dmin)
    closest_node = rrt[ind_min]

    return closest_node

def isCollisionFreeVertex(obj,node):
    node_list = list(node)
    new_obj = []
    
    for x in obj:
        new_obj.append([x.x,x.y,x.z])
    
    if node_list in new_obj:
        return False
    else:
        return True

def rrt_path(obstacles,start,goal):
    
    animate = 1

    # RRT Initialization
    maxiters  = 5000
    nearGoal = False # This will be set to true if goal has been reached
    minDistGoal = 0.05 # Convergence criterion: success when the tree reaches within 0.25 in distance from the goal.
    d = 1 # [m], Extension parameter: this controls how far the RRT extends in each step.

    # Start and goal positions
    #start = np.array(start)    #;ax.scatter3D(start[0], start[1], start[2], color='green', s=100);ax1.scatter3D(start[0], start[1], start[2], color='green', s=100)
    #goal =  np.array(target) #;ax.scatter3D(goal[0], goal[1], goal[2], color='red', s=100);ax1.scatter3D(goal[0], goal[1], goal[2], color='red', s=100)
    # Initialize RRT. The RRT will be represented as a 2 x N list of points.
    # So each column represents a vertex of the tree.
    rrt = []
    start_node = Node3D()
    start_node.p = start
    start_node.i = 0
    start_node.iPrev = 0
    rrt.append(start_node)

    # RRT algorithm
    start_time = time.time()
    iters = 0
    while not nearGoal and iters < maxiters:
    # Sample point
        rnd = random()
        if rnd < 0.10:
            p = goal
        else:
            p = np.array([random()*10, random()*10, random()*10]) # Should be a 3 x 1 vector

        # Check if sample is collision free
        collFree = isCollisionFreeVertex(obstacles, p)
        # If it's not collision free, continue with loop
        if not collFree:
            iters += 1
            continue

        # If it is collision free, find closest point in existing tree. 
        closest_node = closestNode3D(rrt, p)


        # Extend tree towards xy from closest_vert. Use the extension parameter
        # d defined above as your step size. In other words, the Euclidean
        # distance between new_vert and closest_vert should be d.
        new_node = Node3D()
        new_node.p = closest_node.p + d * (p - closest_node.p)
        new_node.i = len(rrt)
        new_node.iPrev = closest_node.i

        #if animate:
            #ax.plot([closest_node.p[0], new_node.p[0]], [closest_node.p[1], new_node.p[1]], [closest_node.p[2], new_node.p[2]],color = 'b', zorder=5)
            #plt.pause(0.01)

        # Check if new vertice is in collision
        collFree = isCollisionFreeVertex(obstacles,new_node.p)
        collFree = isCollisionFreeVertex(obstacles,closest_node.p)
        # If it's not collision free, continue with loop
        if not collFree:
            iters += 1
            continue
        
        # If it is collision free, add it to tree    
        rrt.append(new_node)

        # Check if we have reached the goal
        if norm(np.array(goal) - np.array(new_node.p)) < minDistGoal:
            # Add last, goal node
            goal_node = Node3D()
            goal_node.p = goal
            goal_node.i = len(rrt)
            goal_node.iPrev = new_node.i
            #if isCollisionFreeEdge(obstacles, new_node.p, goal_node.p):
            #    rrt.append(goal_node)
            #    P = [goal_node.p]
            #else: 
            P = []
            end_time = time.time()
            nearGoal = True
            print ('Reached the goal after %.2f seconds:' % (end_time - start_time))

        iters += 1
    i = len(rrt) - 1
    #t = []
    print('p',P)
    while True:
        i = rrt[i].iPrev
        P.append(rrt[i].p)
        if i == 0:
            #print ('Reached RRT start node')
            break
    #print ('Number of iterations passed: %d / %d' %(iters, maxiters))
    #print ('RRT length: ', len(rrt))

    return P


class d3d():
    def __init__(self):
        self.construct_field(objs)
        pass
    def construct_field(self,obj):
        self.field = np.ones((512, 512, 512), dtype=np.int32)
        for i in objs:
            self.field(round.(i[0],round(i[1],round(i[2]))
    pass


def dumb_path():
    x_lim_max = 100
    x_lim_min = 0 
    way_points = [[]]




def main():

    world_agent = world()
    world_agent.models()
    print('Tracando caminho.....')
    
    path = rrt_path(obstacles = world_agent.models_pose ,start = [1,2,0],goal = [8,5,1])
    #print('caminho -- >',path)
    rospy.spin()
    pass

if __name__ == '__main__':

    print("Inicializando....mission_planener")

    rospy.init_node('Mission_planner', anonymous=False)   

    rospy.wait_for_service('/gazebo/get_world_properties')
    
    get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    
    get_models_world = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)

    main()  
	