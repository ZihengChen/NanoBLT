
from Framework import *

class Analyzer():
    def __init__(self):
        temp = __file__.split("/")
        self.anlyzerCodeDir = "/".join(temp[:-1])+"/" 
        self.d_features = {}
        self.enableProfiler = True

        
    def build_cl_program(self):
        clSrc = ""
        # read opencl src files
        for clSrcFile in self.clSrcFiles:
            f = open( self.anlyzerCodeDir + clSrcFile,"r") 
            clSrc += f.read()
            f.close()
        # build opencl program from src
        self.clProgram  = cl.Program(context, clSrc).build()
        # initialize opencl queue on device
        if self.enableProfiler:
            self.clCmdQueue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.clCmdQueue = cl.CommandQueue(context)


    def read_features_from_tree(self, featureConfigs):
        # read tree branch as np array using uproot, which is about 3x faster than root_numpy
        self.features = self.tree.arrays(featureConfigs, flatten=True, namedecode="utf-8")

    def copy_features_to_device(self):
        
        # copy feartures from host to device
        for key in self.features.keys():
            self.d_features[key] = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.features[key] )
            

    def realse_features_on_device(self):
        for key in self.d_features.keys():
            self.d_features[key].release()

    def initiate_features_on_host_and_device(self, featureConfigs):
        # for features
        for key,length,datatype in featureConfigs:
            # initialize place holder on host
            self.features[key] = np.empty(length, dtype=datatype)
            # allocate memory on device, datatype is 4 bytes
            self.d_features[key] = cl.Buffer(context, cl.mem_flags.READ_WRITE, np.nbytes[datatype]*length)      

    def copy_features_to_host(self, featureConfigs):
        # for features
        for key,length,datatype in featureConfigs:
            # copy back to host
            cl.enqueue_copy(self.clCmdQueue, self.features[key], self.d_features[key])



class P4_PtEtaPhiM():
    def __init__(self, pt, eta, phi, m):
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.m = m
        self.px = pt*np.cos(phi)
        self.py = pt*np.sin(phi)
        self.pz = pt*np.sinh(eta)
        self.E = np.sqrt(self.px**2 + self.py**2 + self.pz**2 + self.m**2)
    
    def delta_phi(self, p4):
        return np.abs(self.phi-p4.phi)
    

    def delta_r(self, p4):
        return np.sqrt((self.eta-p4.eta)**2 + (self.phi-p4.phi)**2)
    
    def invariant_mass(self, p4):
        return np.sqrt((self.E+p4.E)**2 - (self.px+p4.px)**2 - (self.py+p4.py)**2 - (self.pz+p4.pz)**2)