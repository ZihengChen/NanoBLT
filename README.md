# NanoBLT
NanoBLT is a `ROOT`-free and `CMSSW`-free analyzer running on [CMS NanoAOD](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD). Its reading of NaonAOD root files is based on [uproot](https://github.com/scikit-hep/uproot), while its event processing is written with [PyOpenCL](https://documen.tician.de/pyopencl/) for high performance parallelization. Its outputs are HDF5 files which is ready-to-plot.

## Example Dilepton Analyzer for Z(ee) Events
An example dilepton analyzer is included for selection Z(ee). In each event, it first select electrons based on kinematics and reconstruction cuts. Then it selects events that pass single electron trigger and have 2 opposite good electrons. 
The performance of the OpenCL kernel is messured with opencl inbuilt profiler. The result on CPUs and GPUs are shown below.

<p align="center">
<img src="https://github.com/ZihengChen/NanoBLT/blob/master/plots/throughputs.png" width="800">
</p>

Dilepton Analyzer saves the selected events as a ready-to-plot dataframe. The plot of dilepton invirant mass is shown below.

<p align="center">
<img src=https://github.com/ZihengChen/NanoBLT/blob/master/plots/Dilepton_mass.png width="500">
</p>


## Setup GPU Interface on LPC
First log into lpc GPU node `ssh -Y username@cmslpcgpu1.fnal.gov`

if you have not installed local anaconda python, do it with
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Downloads/Anaconda3-2020.02-Linux-x86_64.sh -b -u -p /uscms_data/d3/zchen/anaconda
```
Then source CUDA and anaconda environment `source SetupCUDA.sh` and you should be able to see `nvidia-smi`, `nvcc --version` and `which python`. It is helpful to refer the guidance [TWiki](https://twiki.cern.ch/twiki/bin/view/Main/GPUSoftwareInstallConfigure) of LPC GPU. Then install some python package.

```bash
pip install --user uproot pycuda
conda install pyopencl
conda install ocl-icd-system
conda install pocl
```
After doing this, you can try python
```python
import pyopencl as cl
cl.get_plotforms
# should prints the following:
# [ <pyopencl.Platform 'NVIDIA CUDA' at 0x56319edefbb0>, 
#   <pyopencl.Platform 'Portable Computing Language' at 0x7f99c2d7f020>]
``` 
Now, you are all set on LPC. Get NanoBLT and have fun.
```bash
git clone https://github.com/ZihengChen/NanoBLT.git
```
On eos LPC, nanoAOD files of EGamma_Run2018 dataset are located at `/eos/uscms/store/user/zchen/nanoaod/data`





