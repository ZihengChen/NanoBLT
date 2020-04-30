import uproot
import uproot_methods
import numpy as np
import pyopencl as cl
from timeit import default_timer as timer
import warnings
warnings.filterwarnings("ignore")

global device 
device = cl.get_platforms()[0].get_devices()[0]

global context 
context = cl.Context([device])

global blocakSize 
blockSize  = 1024



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