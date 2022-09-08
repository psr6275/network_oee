import argparse
import logging
import os, pickle, sys
import threading


class myThread (threading.Thread):
    def __init__(self, command):
        threading.Thread.__init__(self)
        self.cmd = command

    def run(self):
        print("Starting " + self.cmd)
        os.system(self.cmd)
        print("Exiting " + self.cmd)

def _run_experiments():
    
    default_cmd = "python main_oe.py "
    datasets = ["2017", "2018"]
    thread_list = []
    gid = 2
    for ds in datasets:        
        ds_cmd = default_cmd + "--dataset %s "%ds
        for exp in range(1):
            cmd = ds_cmd + "--device cuda:%s --exp %s"%(exp+1, exp+1)
            thread_list.append(myThread(cmd))
        cmd = ds_cmd + "--device cuda:%s --VAN"%(gid)
        gid +=1
        thread_list.append(myThread(cmd))
    for thr in thread_list:
        thr.start()

if __name__ == "__main__":
    _run_experiments()