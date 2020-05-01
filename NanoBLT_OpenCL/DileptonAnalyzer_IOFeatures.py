
# @author Ziheng Chen
# @email zihengchen2015@u.northwestern.edu
# @create date 2020-05-01 00:40:35
# @modify date 2020-05-01 00:40:35

import pandas as pd
from Framework import *
from Analyzer import P4_PtEtaPhiM

def load_features_from_nanoaod(self, nanoaodFile):
    # parse file dir, name, suffix
    temp = nanoaodFile.split("/")
    self.nanoaodFilePath = "/".join(temp[:-1])+"/" 
    self.nanoaodFileName, self.nanoaodFileSuffix = temp[-1].split(".")

    # ========== input features ==========
    self.inputFeatureConfigs = [
        # object features
        'Electron_pt',              #float32
        'Electron_eta',             #float32
        'Electron_phi',             #float32
        'Electron_mass',            #float32
        'Electron_cutBased',        #int32
        'Electron_pfRelIso03_all',  #float32
        'Electron_pdgId',           #int32

        # event features
        'HLT_Ele32_WPTight_Gsf', #bool
        'nElectron',    #int32
    ]

    start = timer()
    self.tree = uproot.open(nanoaodFile)["Events"]
    # read root file into array
    self.read_features_from_tree(self.inputFeatureConfigs)
    # precomput object indexing
    self.features['nElectron_headIndex'] = np.insert(np.cumsum(self.features['nElectron'])[:-1],0,0).astype(np.int32)
    self.n = np.int32(self.features['HLT_Ele32_WPTight_Gsf'].size)

    end = timer()
    # debug info
    if self.verboseRunningInfo:
        print("--- n = ",self.n)
        print("io read : {:10.2f} ms".format((end-start)*1e3))


def initiate_output_features_on_host_and_device(self):
    
    # ========== output features ==========
    self.outputFeatureConfigs = [
        # no object features are allowed to output
        # event features
        ("Event_catagory",self.n, np.int32),
        ("nPassElectron",self.n, np.int32),

        ("Lepton1_pt", self.n, np.float32),
        ("Lepton1_eta", self.n, np.float32),
        ("Lepton1_phi", self.n, np.float32),
        ("Lepton1_mass", self.n, np.float32),
        ("Lepton1_pdgId", self.n, np.int32),
        ("Lepton1_reliso", self.n, np.float32),
        
        ("Lepton2_pt", self.n, np.float32),
        ("Lepton2_eta",self.n, np.float32),
        ("Lepton2_phi",self.n, np.float32),
        ("Lepton2_mass", self.n, np.float32),
        ("Lepton2_pdgId", self.n, np.int32),
        ("Lepton2_reliso", self.n, np.float32),

    ]
    self.initiate_features_on_host_and_device(self.outputFeatureConfigs)


def save_catagorized_events_as_h5(self, outputDir=None):
    if not outputDir:
        outputDir = self.nanoaodFilePath
    
    if not os.path.exists(outputDir):
        os.system('mkdir -p '+outputDir)

    start = timer()
    for catagory in range(1):
        filter = self.features['Event_catagory'] == catagory
        df = pd.DataFrame()
        for key,_,__ in self.outputFeatureConfigs:
            df[key] = self.features[key][filter]
        # post process dataframe
        self.postProcessDataFrame(df)
        # save it as h5
        df.to_hdf(outputDir + self.nanoaodFileName + ".h5", key=str(catagory))
    # reset features
    self.features = None
    end = timer()
    if self.verboseRunningInfo: 
        print("io write: {:10.2f} ms".format((end-start)*1e3))

def postProcessDataFrame(self, df):
    lep1 = P4_PtEtaPhiM(df.Lepton1_pt, df.Lepton1_eta, df.Lepton1_phi, df.Lepton1_mass)
    lep2 = P4_PtEtaPhiM(df.Lepton2_pt, df.Lepton2_eta, df.Lepton2_phi, df.Lepton2_mass)
    df["Dilepton_mass"] = lep1.invariant_mass(lep2)
    df["Leptons_deltaR"] = lep1.delta_r(lep2)
    df["Leptons_deltaPhi"] = lep1.delta_phi(lep2)


