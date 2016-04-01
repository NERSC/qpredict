import os
import sys

import data
from queues import *


def main():
	"""Driver routine"""
	# Global params
	dir = "/global/homes/a/ankitb/qpredictor/data/"
	tstart = "1458591001"
	db = "staffdb01"
	user = "usgweb_ro"
	passwd = "rHpJ1ZdEij8="
	table = "jobs"
	
	for machine in ['cori']:
		debug = DebugQueue(machine)
		regular = RegQueue(machine)
		shared = SharedQueue(machine)
	 
		queue = {} # Return object for queue data
		completed = {} # Return object for completed job data
	
		try:
			with open("snapshots_"+machine+".txt") as f:
				timestamps = f.read().splitlines()
		except IOError:
			print "Error opening timestamps file"
	
		# Get current queue data
		queue = data.loadQueuedJobData(machine,dir,timestamps)
		
		debug.queuedJobs  = {k:v for k,v in queue.items() if v.partition == 'debug'} #filter(lambda x: x.partition == 'debug', coriJobList)
		regular.queuedJobs = {k:v for k,v in queue.items() if (v.partition == 'regular') | (v.partition == 'regularx')} 
		shared.queuedJobs = {k:v for k,v in queue.items() if v.partition == 'shared'} #filter(lambda x: x.partition == 'shared', coriJobList) 
	
		# Get completed jobs data
		completed = data.loadCompletedJobData(machine,tstart,db,user,passwd,table)
	
	
		# Form input and output vectors to pass to ML routine

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
