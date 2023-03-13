import h5py
import numpy as np

datasetFile = '/home/juan/GRVC/uzh-rpg/IJRR2017/boxes/boxes_6dof/boxes_6dof.hdf5'
# datasetFile = '/home/juan/GitHub/tub-rip/event_based_optical_flow/datasets/MVSEC/hdf5/indoor_flying4_data.hdf5'
# datasetFile = '/home/juan/GRVC/uzh-rpg/IJRR2017/boxes/boxes_6dof/test/boxes_6dof.h5'
# datasetFile = '/home/juan/GitHub/tub-rip/event_based_optical_flow/datasets/MVSEC/hdf5/indoor_flying4_data.hdf5'
data = h5py.File(datasetFile, "r")

print("Keys: %s" % data.keys())
print("Keys: %s" % data["davis"].keys())


# print("Keys: %s" % data["davis"])
# print("Keys: %s" % data["davis"]["left"])
# print(np.shape(data["davis"]["left"]["events"]))
# x = np.array(data["davis"]["right"]["events"][:,3])
# y = np.array(data["davis"]["right"]["events"][:,1])
ts = np.array(data["davis"]["image_raw_ts"])
print(np.shape(ts))
