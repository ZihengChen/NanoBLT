
# @author Ziheng Chen
# @email zihengchen2015@u.northwestern.edu
# @create date 2020-05-01 00:42:30
# @modify date 2020-05-01 00:42:30


import uproot
import uproot_methods
import numpy as np
import pyopencl as cl
import os,sys
from timeit import default_timer as timer
import warnings
warnings.filterwarnings("ignore")

global device 
device = cl.get_platforms()[0].get_devices()[0]

global context 
context = cl.Context([device])

global blocakSize 
blockSize  = 1024 #device.max_work_group_size
