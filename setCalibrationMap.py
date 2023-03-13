import cv2
import numpy as np

height = 260
width = 346

def load_map_txt(map_txt):
    f = open(map_txt, "r")
    line = f.readlines()
    map_array = np.zeros((height, width))
    for i, l in enumerate(line):
        map_array[i] = np.array([float(k) for k in l.split()])
    return map_array

pathToMapX = '/home/juan/GitHub/tub-rip/event_based_optical_flow/datasets/MVSEC/hdf5/indoor_flying_left_x_map.txt'
pathToMapY = '/home/juan/GitHub/tub-rip/event_based_optical_flow/datasets/MVSEC/hdf5/indoor_flying_left_y_map.txt'

K = np.array([[226.38018519795807, 0.0, 173.6470807871759],
                         [0.0, 226.15002947047415, 133.73271487507847],
                         [0.0, 0.0, 1.0]])
D = np.array([[-0.048031442223833355], [0.011330957517194437], [-0.055378166304281135], [0.021500973881459395]])

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (width,height), cv2.CV_16SC2)

point = np.zeros((1,1,2), dtype=np.float32)
undistortPoint = cv2.fisheye.undistortPoints(point, K, D)
# x_map = np.load(datasetFile)
# print(np.shape(x_map))

map_array_x = load_map_txt(pathToMapX)
map_array_y = load_map_txt(pathToMapY)

print(map_array_x)
print(map_array_y)
print((undistortPoint))
