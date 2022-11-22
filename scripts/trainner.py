import rospy

from std_msgs.msg import Bool


def main():
    
    for i in range(5):
        reset.publish(Bool(data = True))
    
    rospy.spin()

if __name__ == '__main__':
    
    rospy.init_node('trainner')

    reset = rospy.Publisher('/gazebo/reset_uav_pose',Bool,queue_size=1)

    main()      
