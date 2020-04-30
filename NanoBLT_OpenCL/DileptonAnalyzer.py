from Analyzer import *


class DileptonAnalyzer(Analyzer):
    from DileptonAnalyzer_IOFeatures import \
        load_features_from_nanoaod, \
        initiate_output_features_on_host_and_device, \
        save_event_retures_as_h5, \
        postProcessDataFrame


    def __init__(self):
        super().__init__()
        self.clSrcFiles = [
            'DileptonAnalyzer.c'
        ]
        self.build_cl_program()
        self.verboseRunningInfo = True
        print(device.platform.name, device.name, device.version)
    

    def run_analyzer(self):
        
        start = timer()
        # copy input to device
        self.copy_features_to_device()
        # run opencl functions
        self.clProgram.run_analyzer(
            # opencl kernal configration, (queue, globalSize, localSize)
            self.clCmdQueue, ((int(self.n/blockSize)+1)*blockSize,), (int(blockSize),),
            
            # ========== output ==========
            # no object features are allowed to output
            # event features
            self.d_features['Event_catagory'],  #int32
            self.d_features['nPassElectron'],   #int32

            self.d_features['Lepton1_pt'],      #float32
            self.d_features['Lepton1_eta'],     #float32
            self.d_features['Lepton1_phi'],     #float32
            self.d_features['Lepton1_mass'],    #float32
            self.d_features['Lepton1_pdgId'],   #int32
            self.d_features['Lepton1_reliso'],  #float32

            self.d_features['Lepton2_pt'],      #float32
            self.d_features['Lepton2_eta'],     #float32
            self.d_features['Lepton2_phi'],     #float32
            self.d_features['Lepton2_mass'],    #float32
            self.d_features['Lepton2_pdgId'],   #int32
            self.d_features['Lepton2_reliso'],  #float32

            # ========== input ==========
            # object features
            self.d_features['Electron_pt'],             #float32
            self.d_features['Electron_eta'],            #float32
            self.d_features['Electron_phi'],            #float32
            self.d_features['Electron_mass'],           #float32
            self.d_features['Electron_cutBased'],       #int32
            self.d_features['Electron_pfRelIso03_all'], #float32
            self.d_features['Electron_pdgId'],          #int32

            # event features
            self.d_features['HLT_Ele32_WPTight_Gsf'], #bool
            self.d_features['nElectron'], self.d_features['nElectron_headIndex'], #int32
            
            # ========== constants ==========  
            self.n
        )
        # copy output features to host
        self.copy_features_to_host(self.outputFeatureConfigs)
        end = timer()
        if self.verboseRunningInfo:
            print("evtCata: ",end-start)