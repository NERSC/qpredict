import os
import sys
# Local imports
import data
import test
import argparse
from queues import *

import json

import numpy as np
import pandas as pd

def main():
	"""Driver routine"""
	#parse arguments
	parser = argparse.ArgumentParser(description='Collect data for training.')
	parser.add_argument('--inputfile', type=str, nargs=1, default=None, help='Specify the location of the json descriptor file.')
	
	#parse arguments
	args = parser.parse_args()
	inputjson=args.inputfile[0]
	if not inputjson:
		raise ValueError('Please specify a valid json input file with --inputfile <filename>.')
	
	# parse the parameter input file
	with open(inputjson) as params:
		input = json.load(params)

	base_dir = input['base_dir']
	temp_dir = input['temp_dir']
	data_dir = input['data_dir']
	output_dir = input['output_dir']
	db_cred_file = input['db_cred_file']
	machines = input['machines']
	tstart = input['tstart']
	tend = input['tend']
	one_hot = bool(input['one_hot'])
	train_fraction = float(input['train_fraction'])
	validation_fraction = float(input['validation_fraction'])
	test_fraction = float(input['test_fraction'])
	
	for machine in machines:
		#get queue information
		queue,completed = data.get_data(machine,base_dir,temp_dir,data_dir,db_cred_file,tstart,tend)
		hotdf = data.create_df(queue,completed,one_hot)
		test.create_all_sets(output_dir,hotdf,train_fraction,validation_fraction,test_fraction) 

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
