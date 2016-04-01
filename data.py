import os
import sys
import MySQLdb

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


def loadCompletedJobData(machine,start,db,user,passwd,table):
	"""Loads completed job data from staffdb01. One caveat is that
	jobs that did not "finish" are not stored so there may be a mismatch
	in the jobs listed in queue snapshots and the completed jobs list 
	returned by this function 
	"""
	# Open database connection
	#mdb = MySQLdb.connect("staffdb01","usgweb_ro","rHpJ1ZdEij8=","jobs" )
	mdb = MySQLdb.connect(db,user,passwd,table)
	
	# prepare a cursor object using cursor() method
	cursor = mdb.cursor()
	
	t0 = start 
	
	sql_cmd="SELECT stepid, numnodes, class, dispatch, start, completion, wallclock, wait_secs, superclass, wallclock_requested FROM summary WHERE hostname='"+machine+"' AND start>"+start
	
	# execute SQL query using execute() method.
	cursor.execute(sql_cmd)
	
	finishedJobs = {}
	
	# Fetch a single row using fetchone() method.
	for data in cursor.fetchall():
	    jobId = int(data[0].split('.')[0])
	    qos = data[2]
	    partition = data[8]
	    reqWalltime = data[6]
	    obsWalltime = data[5]-data[4]
	    reqNodes = data[1]
	    obsWaitTime = data[7]
	
	    finishedJobs[jobId] = CompletedJob(machine,jobId,partition,qos,reqWalltime,reqNodes,obsWalltime,obsWaitTime)
	
	# disconnect from server
	mdb.close()
	
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
