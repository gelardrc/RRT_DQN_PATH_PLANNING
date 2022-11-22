import rospy
import random
import keyboard
from pcg_gazebo.simulation import create_object
from pcg_gazebo.generators import WorldGenerator

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan,PointCloud2


class Drone(object):

    
    def __init__(self):
       ## inicializa os sensores ##
       self.sensores = Sensors()
       ## Inicializa os subscribers do vant ##
       self.sub()

       pass
    
    def get_pose_cb(self,state):
        self.x,self.y,self.z = [state.pose.position.x,state.pose.position.y,state.pose.position.z] 
        pass
    
    def sub(self):
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped,self.get_pose_cb)
        pass

class Sensors(object):
    def __init__(self):
        self.lidar = Lidar()
        self.camera = Camera()
        pass
    
class Lidar(object):
    def __init__(self):
        rospy.Subscriber('/agent/lidar', LaserScan,self.get_readings_cb)
    def get_readings_cb(self,readings):
        self.readings = readings.ranges
        #print(self.readings)
class Camera(object):
    def __init__(self):
        rospy.Subscriber('/agent/camera/depth/points', PointCloud2,self.get_depth_cb)
        pass
    def get_depth_cb(self,readings):
        self.depth = readings
        #print(self.depth)
        pass

def main():

   rospy.spin()

   pass
 

if __name__ == '__main__':

    print("Inicializando Agente....")

    drone = Drone()

    rospy.init_node('Agent', anonymous=False)   
    
    
    main()  
	
