#!/usr/bin/env python
# coding: utf-8


from DileptonAnalyzer import *
import glob
import threading, queue
inputDir  = "/home/zchen/cms/data/EGamma_Run2018B/"
outputDir = "/home/zchen/Documents/Analysis/nanoaod/data/hdf5/EGamma_2018B/"
fileNames = glob.glob(inputDir+"*.root")


nWorkers = 8
jobQueue = queue.Queue()
def worker():
    jobId, files = jobQueue.get()
    ana = DileptonAnalyzer()
    for file in files:
        ana.load_features_from_nanoaod(file)
        ana.initiate_output_features_on_host_and_device()
        ana.run_analyzer()
        ana.realse_features_on_device()
        ana.save_catagorized_events_as_h5(outputDir)
    jobQueue.task_done()

 
# turn-on the worker thread
for i in range(nWorkers):
    threading.Thread(target=worker, daemon=True).start()

start = timer()  
# send thirty task requests to the worker
for jobId, partialFileNames in enumerate([fileNames[i::nWorkers] for i in range(nWorkers)]):
    jobQueue.put((jobId, partialFileNames))
    
print('All job requests sent\n', end='')

# block until all tasks are done
jobQueue.join()
end = timer() 
print('All jobs are completed. total time: ',end-start)

