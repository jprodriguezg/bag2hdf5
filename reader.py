import h5py
import numpy as np
from utils.eventslicer import EventSlicer

datasetFile = '/home/juan/GitHub/tub-rip/event_based_optical_flow/datasets/DAVIS/hdf5/boxes_6dof_data.hdf5'
data = h5py.File(datasetFile, "r")

print("Keys: %s" % data.keys())
print("Keys: %s" % data["davis"].keys())
ts = np.array(data["davis"]["image_raw_ts"], dtype=np.float64)
