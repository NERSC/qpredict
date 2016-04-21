import os
import sys
# Local imports
import data
import test
from queues import *

import json

import numpy as np
import pandas as pd

def main():
    """Driver routine"""
    # Global params
    with open('input/params.json') as params:
    	input = json.load(params)

	base_dir = input['base_dir']
	data_dir = input['data_dir']
	db_cred_file = input['db_cred_file']
	machines = input['machines']
	tstart = input['tstart']
	tend = input['tend']
	one_hot = bool(input['one_hot'])
	train_fraction = float(input['train_fraction'])
	validation_fraction = float(input['validation_fraction'])
	test_fraction = float(input['test_fraction'])
    
    for machine in machines:
		debug = DebugQueue(machine)
		regular = RegQueue(machine)
		shared = SharedQueue(machine)
	
		queue,completed = data.get_data(machine,base_dir,data_dir,db_cred_file,tstart,tend)
		hotdf = data.create_df(queue,completed,one_hot)
		test.create_all_sets(hotdf,train_fraction,validation_fraction,test_fraction) 

def sanity_check(check_flag):		
		if check_flag == 'print_obs':	
			for job,attrib in regular.queuedJobs.items():
				if job in completed:
					print completed[job].obsWalltime, completed[job].obsWaitTime
	
		if check_flag == 'list_lengths':
			print len(debug.queuedJobs)
			print len(regular.queuedJobs)
			print len(shared.queuedJobs)
	
		if check_flag == 'print_completed_queue':
			for k,v in completed.items():
				print v.machine,v.jobId,v.partition,v.qos,v.reqWalltime,v.reqNodes,v.obsWalltime,v.obsWaitTime

if __name__ == '__main__':
	main()	
