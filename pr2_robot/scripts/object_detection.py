#!/usr/bin/env python

# Import modules
import os 

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

#test case, 1, 2, 3 for  world1, world2, and world3
CASE_NUM = 3


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    # print(dict_list)
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.005
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # PassThrough Filter
    passthrough_z = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough_z.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 0.9
    passthrough_z.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough_z.filter()

    passthrough_y = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough_y.set_filter_field_name(filter_axis)
    axis_min = -0.4
    axis_max = 0.4
    passthrough_y.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough_y.filter()
    # RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)
    # Statistical outlier removal filter
    outlier_filter = extracted_outliers.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(20)

    # Set threshold scale factor
    x = 0.1

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()



    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_filtered)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(3000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    detected_objects_labels = []
    detected_objects = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])
        object_cluster_cloud = cloud_filtered.extract(indices)

        #convert from pcl point cloud to ros msg
        ros_object_cluster_cloud = pcl_to_ros(object_cluster_cloud)
        #publish 
        obj_original_color_pub.publish(ros_object_cluster_cloud)
        #compute the feature for this object
        color_features = compute_color_histograms(ros_object_cluster_cloud, using_hsv=True)
        object_normal = get_normals(ros_object_cluster_cloud)
        normal_features = compute_normal_histograms(object_normal)
        object_features = np.concatenate((color_features, normal_features))
        #make prediction using the trained model and add label
        pred = clf.predict(scaler.transform(object_features.reshape(1, -1)))
        label = encoder.inverse_transform(pred)[0]
        detected_objects_labels.append(label)
        #publish to rviz of this object
        # Publish a label into RViz
        label_pos = list(white_cloud[indices[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, j))
         # Add the detected object
        do = DetectedObject()
        do.label = label
        do.cloud = ros_object_cluster_cloud
        detected_objects.append(do)
    #output result of recognition
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    #visualize segmentation in rviz
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    # Convert PCL data to ROS messages
    segment_ros_points = pcl_to_ros(cluster_cloud)
    table_object_ros_points = pcl_to_ros(cloud_filtered)
    # Publish ROS messages
    obj_segment_pub.publish(segment_ros_points) #the segmented object
    obj_pub.publish(table_object_ros_points) #the table object after filtering
    detected_objects_pub.publish(detected_objects) #the detected object

    #move the object to matched dropbox
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    #Initialize variables
    TEST_SCENE_NUM = Int32()
    TEST_SCENE_NUM.data = CASE_NUM
    OBJECT_NAME = String()
    WHICH_ARM = String()
    PLACE_POSE = Pose()
    PICK_POSE = Pose()
    #output
    res = []


    # Get object list parameter and dropbox parameter
    object_list_param = rospy.get_param('/object_list') #get object list in the environment
    dropbox_param = rospy.get_param('/dropbox') #get the dropbox parameter


    #make dropbox information
    dropbox_group_name = {} #key is group, val is name
    dropbox_group_pos = {} #key is group, val is position
    for box in dropbox_param:
        dropbox_group_name[box['group']] = box['name']
        dropbox_group_pos[box['group']] = box['position']

    for object_param in object_list_param:
        name = object_param['name']
        group = object_param['group']
        found = False
        #pick this object if we detected
        for index, detected_obj in enumerate(object_list):
            if detected_obj.label != name: continue
            #fill OBJECT_NAME
            OBJECT_NAME.data = detected_obj.label
            #fill WHICH_ARM
            WHICH_ARM.data = dropbox_group_name[group]
            #fill PICK_POSE 
            #convert cloud data of detected objet to pcl then numpy array
            cloud_pts_arr = ros_to_pcl(detected_obj.cloud).to_array()
            #calculate the centroid of these points
            centroids = np.mean(cloud_pts_arr, axis=0)[:3]
            #convert to python float64
            centroids = [np.asscalar(x) for x in centroids]
            PICK_POSE.position.x = centroids[0]
            PICK_POSE.position.y = centroids[1]
            PICK_POSE.position.z = centroids[2]
            #fill PLACE_POSE
            box_pos = dropbox_group_pos[group]
            PLACE_POSE.position.x = box_pos[0]
            PLACE_POSE.position.y = box_pos[1]
            PLACE_POSE.position.z = box_pos[2]

            #Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
            object_yaml = make_yaml_dict(TEST_SCENE_NUM, WHICH_ARM,
                                         OBJECT_NAME, PICK_POSE,
                                         PLACE_POSE)
            res.append(object_yaml)
            print("Handle {}".format(name))
            found = True
            del object_list[index]
            break
        # Wait for 'pick_place_routine' service to come up
        if not found: continue
        rospy.wait_for_service('pick_place_routine')
        if not found: continue
        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            #Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)
            if resp.success: break

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    # send_to_yaml('output_1.yaml', res)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node("object_recog", anonymous=True)

    # Create Subscribers
    point_cloud_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    #Create Publishers
    obj_pub = rospy.Publisher('/table_obj', 
                              PointCloud2, 
                              queue_size=1)
    obj_segment_pub = rospy.Publisher('/obj_segment', 
                                      PointCloud2, 
                                      queue_size=1)
    object_markers_pub = rospy.Publisher('/obj_recognition', 
                                         Marker, 
                                         queue_size=1)
    detected_objects_pub = rospy.Publisher('/detected_objects',
                                           DetectedObjectsArray,
                                           queue_size=1)
    obj_original_color_pub = rospy.Publisher('/object_original_color', 
                                             PointCloud2, 
                                             queue_size=1)
    # Load model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    model_items = pickle.load(open('model_200.sav', 'rb'))
    clf, classes, scaler = model_items['classifier'], model_items['classes'], model_items['scaler']
    encoder = LabelEncoder()
    encoder.classes_ = classes

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
