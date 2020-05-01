# NanoBLT
NanoBLT is a `ROOT`-free and `CMSSW`-free analyzer running on [CMS NanoAOD](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD). Its reading of NaonAOD root files is based on [uproot](https://github.com/scikit-hep/uproot), while its event processing is written with [PyOpenCL](https://documen.tician.de/pyopencl/) for high performance parallelization. Its outputs are HDF5 files which is ready-to-plot.

## Example Dilepton Analyzer for Z(ee) Events
An example dilepton analyzer is included for selection Z(ee). In each event, it first select electrons based on kinematics and reconstruction cuts. Then it selects events that pass single electron trigger and have 2 opposite good electrons. 
The performance of the OpenCL kernal is messured with opencl inbuilt profiler. The result on CPUs and GPUs are shown below.

<p align="center">
<img src="https://github.com/ZihengChen/NanoBLT/blob/master/plots/throughputs.png" width="800">
</p>

Dilepton Analyzer saves the selected events as a ready-to-plot dataframe. The plot of dilepton invirant mass is shown below.

<p align="center">
<img src=https://github.com/ZihengChen/NanoBLT/blob/master/plots/Dilepton_mass.png width="500">
</p>
