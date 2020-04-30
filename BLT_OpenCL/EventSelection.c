// objectSelection_muons
# define MAXNLEPTON 3
# define MAXNJET 5

__kernel void run_analyzer(
    // ========== output ==========
    // no object features are allowed to output
    // event features
    __global int *Event_catagory,
    __global int *nPassElectron,

    __global float *Lepton1_pt,
    __global float *Lepton1_eta,
    __global float *Lepton1_phi,
    __global float *Lepton1_mass,
    __global int *Lepton1_pdgId,
    __global float *Lepton1_reliso,

    __global float *Lepton2_pt,
    __global float *Lepton2_eta,
    __global float *Lepton2_phi,
    __global float *Lepton2_mass,
    __global int *Lepton2_pdgId,
    __global float *Lepton2_reliso,

    // ========== input ==========  
    // object features
    __global float *Electron_pt,
    __global float *Electron_eta,
    __global float *Electron_phi, 
    __global float *Electron_mass,
    __global int *Electron_cutBased,
    __global float *Electron_pfRelIso03_all,
    __global int *Electron_pdgId, 

    // event features
    __global bool *HLT_Ele32_WPTight_Gsf,
    __global int *nElectron, __global int *nElectron_headIndex,

    // ========== constants ==========  
    const int n
    

) {
    int i = get_group_id(0)*get_local_size(0)+get_local_id(0);
    if(i < n) {
        // -------------------
        // object selection
        // -------------------
        // electrons
        // loop over all enectrons in the event
        const int ieHeadIndex = nElectron_headIndex[i], ieLength = nElectron[i];
        int ie_pass[MAXNLEPTON];
        int ie_pass_len = 0;
        for( int ie = ieHeadIndex; ie < ieHeadIndex+ieLength; ie++ ){
            if (ie_pass_len >= MAXNLEPTON) break;

            if( Electron_pt[ie]>15
                && -2.5<Electron_eta[ie]<2.5
                && Electron_cutBased[ie]>=3
            ){
                ie_pass[ie_pass_len] = ie;
                ie_pass_len++;
            }
        }
        // -------------------
        // event selection
        // -------------------
        int eventCatagory = -1;
        if( ie_pass_len >= 2 ) { // ee events
            eventCatagory = 0;

            // cut on trigger
            if (!HLT_Ele32_WPTight_Gsf[i])
                eventCatagory = -1;

            // cut on leading and trailing pt
            int ilep1=ie_pass[0], ilep2=ie_pass[1];
            if( Electron_pt[ilep1]<32 || Electron_pt[ilep2]<15)
                eventCatagory = -1;

            // cut on opposite sign
            if(Electron_pdgId[ilep1] * Electron_pdgId[ilep2] > 0)
                eventCatagory = -1;

            if (eventCatagory>=0){
                // save lepton info
                nPassElectron[i] = ie_pass_len;

                Lepton1_pt[i]  = Electron_pt[ilep1];
                Lepton1_eta[i] = Electron_eta[ilep1];
                Lepton1_phi[i] = Electron_phi[ilep1]; 
                Lepton1_mass[i] = Electron_mass[ilep1];   
                Lepton1_pdgId[i] = Electron_pdgId[ilep1]; 
                Lepton1_reliso[i] = Electron_pfRelIso03_all[ilep1];   

                Lepton2_pt[i]  = Electron_pt[ilep2];
                Lepton2_eta[i] = Electron_eta[ilep2];
                Lepton2_phi[i] = Electron_phi[ilep2]; 
                Lepton2_mass[i] = Electron_mass[ilep2];   
                Lepton2_pdgId[i] = Electron_pdgId[ilep2]; 
                Lepton2_reliso[i] = Electron_pfRelIso03_all[ilep2]; 
            }

        } 
        Event_catagory[i] = eventCatagory;
    
    }
}
