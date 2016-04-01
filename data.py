import os
import sys

#dbstuff
from sqlalchemy import (create_engine, inspect, desc, Table, Column, Integer, String, DateTime, MetaData)
from sqlalchemy.sql import select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json

from queues import *
import aux

def loadQueuedJobData(machine,dir,timestamps):
	"""Loads jobs data from stored snapshots. One subtlety here 
	is that the sprio snapshot needs to be read first since it only 
	stores jobs that are actually in queued state. The sqs output 
	also contains jobs that are in states other than queued and those
	need to be excluded for our purpose. The returned object contains 
	a list of unique jobs in the queue from a given interval. 		
	"""
	# Temp dict object
	tempJobs = {}
	
	for time in timestamps:
		snapFileName = dir + "snapshot." + time 
		prioFileName = dir + "priority-factors." + time 
	
		#Unzip the file we need
		os.system("gunzip" + " " + snapFileName + ".gz")
		os.system("gunzip" + " " + prioFileName + ".gz")
	
		count = 0
		uniqueJobsInThisSnapshot = []
		# Do stuff with the data
		with open(prioFileName) as pf:
			pf.readline()
			for line in pf.readlines():
				count += 1
				jobs = line.split()
				jobId = int(jobs[0])
				priority = int(jobs[2])
				age = int(jobs[3])
				fairshare = int(jobs[4])
				qos_int = int(jobs[7])
				rank_p = count
				if jobId not in tempJobs:
					tempJobs[jobId] = QueuedJob(machine,jobId,None,None,None,None,priority,age,fairshare,qos_int,rank_p)
					uniqueJobsInThisSnapshot.append(jobId)
	
		with open(snapFileName) as sf:
			sf.readline()	
			for line in sf.readlines():
				jobs = line.split()
				jobId = int(jobs[0])
				if jobId in uniqueJobsInThisSnapshot:
					tempJobs[jobId].partition = jobs[3]
					tempJobs[jobId].qos = jobs[4].split('_')[0]
					tempJobs[jobId].reqNodes = int(jobs[5])	
					tempJobs[jobId].reqWalltime = aux.convertWalltimeToSecs(jobs[7])
					tempJobs[jobId].priority = int(jobs[8])
	
		#Done reading and setting up full machine queue
	
		# Zip it back up
		os.system("gzip" + " " + snapFileName)
		os.system("gzip" + " " + prioFileName)
	
	# Exotic queues
	#tmplist = {k:v for k,v in tempJobs.items() if v.partition != 'shared' and v.partition != 'regular' and v.partition != 'debug'} #filter(lambda x: x.partition != 'shared' and x.partition != 'regular' and x.partition != 'debug' , coriJobList)

	return tempJobs


def loadCompletedJobData(machine,start,db,user,passwd,hostname,port):
	"""Loads completed job data from staffdb01. One caveat is that
	jobs that did not "finish" are not stored so there may be a mismatch
	in the jobs listed in queue snapshots and the completed jobs list 
	returned by this function 
	"""

	#create engine
	eng = create_engine('mysql://'+user+':'+passwd+'@'+hostname+':'+port+'/'+db)
	base = declarative_base()
	base.metadata.bind = eng
	base.metadata.reflect(bind=eng)
	
	#start session
	session = sessionmaker(bind=eng)
	sess = session()
	
	t0 = start 
	
	summary=base.metadata.tables['summary']
	query=sess.query(
		             summary.c['stepid'],
		             summary.c['numnodes'],
		             summary.c['class'],
		             summary.c['dispatch'],
		             summary.c['start'],
		             summary.c['completion'],
		             summary.c['wallclock'],
		             summary.c['wait_secs'],
		             summary.c['superclass'],
		             summary.c['wallclock_requested']
		             ).filter(
		             summary.c.start>=start,
		             summary.c.hostname==machine
		             )
	
	finishedJobs = {}
	
	# Fetch a single row using fetchone() method.
	for data in query.all():
		jobId = int(data[0].split('.')[0])
		qos = data[2]
		partition = data[8]
		reqWalltime = data[6]
		obsWalltime = data[5]-data[4]
		reqNodes = data[1]
		obsWaitTime = data[7]
									
		finishedJobs[jobId] = CompletedJob(machine,jobId,partition,qos,reqWalltime,reqNodes,obsWalltime,obsWaitTime)
							
    # close session
	sess.close()
			
	#return finished jobs as a list of CompletedJob instances
	return finishedJobs


# Debug/test
if __name__ == '__main__':
	print len(coriDebugQ.queuedJobs)
	print len(coriRegQ.queuedJobs)
	print len(coriSharedQ.queuedJobs)

	#for k,v in tmplist.items():
	#    print v.getPartition()

	#for k,v in coriJobs.items():
	#	if v.partition == 'shared':
	#		print v.machine,v.jobId,v.partition,v.qos,v.reqWalltime,v.reqNodes,v.priority,v.age,v.fairshare,v.qos_int,v.rank_p
else:
	print "data module loaded"
