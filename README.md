# NanoBLT
NanoBLT is a `ROOT`-free and `CMSSW`-free analyzer running on CMS NanoAOD. Its reading of NaonAOD file is based on `uproot`, while its event processing is based on `pyOpenCL`. Its outputs are HDF5 files which is ready-to-plot.

## Example Dilepton Analyzer for Z(ee) Events
An example dilepton analyzer is included for selection Z(ee). In each event, it first select electrons based on kinematics and reconstruction cuts. Then it selects events that pass single electron trigger and have 2 opposite good electrons. 
The performance of the OpenCL kernal is messured with opencl inbuilt profiler. The result on CPUs and GPUs are shown below.

<p align="center">
<img src="https://github.com/ZihengChen/NanoBLT/blob/master/plots/kernalThroughput.png" width="600">
</p>

Dilepton Analyzer saves the selected events as a ready-to-plot dataframe. The plot of dilepton invirant mass is shown below.

<p align="center">
<img src=https://github.com/ZihengChen/NanoBLT/blob/master/plots/Dilepton_mass.png width="400">
</p>
