import os
import sys

import numpy as np
import pandas as pd
import json

#dbstuff
from sqlalchemy import (create_engine, inspect, desc, Table, Column, Integer, String, DateTime, MetaData)
from sqlalchemy.sql import select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import aux
import data
from queues import *

def get_data(machine,base_dir,data_dir,db_cred_file,tstart,tend):
	"""	Creates a new test set independent from one used in training/validation/test sequence """
	
	snapshot_file = base_dir+"input/snapshots_"+machine+".txt"

	print 'Opening snapshots file'
	try:
		with open(snapshot_file) as f:
			timestamps_all = f.read().splitlines()
	except IOError:
		print "Error opening timestamps file"
	print 'Done'

	# For completed jobs database
	with open(db_cred_file) as mysql_creds:
		conn_config = json.load(mysql_creds)

	hostname=conn_config['host']
	databasename=conn_config['database']
	username=conn_config['user']
	password=conn_config['password']
	portnumber=conn_config['port']

	istart = timestamps_all.index(tstart)
	iend = timestamps_all.index(tend)

	timestamps = timestamps_all#[istart:iend]

	queue = {}
	completed = {}

	print 'Loading queue data'
	# Get current queue data
	queue = data.loadQueuedJobData(machine,data_dir,timestamps)
	print 'Done'
	
	print 'Loading completed jobs data'
	# Get completed jobs data
	completed = data.loadCompletedJobData(machine,tstart,tend,databasename,username,password,hostname,portnumber)	
	print 'Done'

	return queue, completed

def create_df(queue, completed, one_hot):
	"""Packs queued and completed job data in dataframes"""
	#Pack it in dataframes
	queuedf=pd.DataFrame([x.to_dict() for x in queue.values()])
	#rename columns with class labels to apply one-hot later
	queuedf.rename(columns={'partition':'partition_tag','qos':'qos_tag'},inplace=True)
	queuedf.sort_values(by=['jobId'],inplace=True)

	#completed
	compdf=pd.DataFrame([x.to_dict() for x in completed.values()])
	del compdf['machine']
	del compdf['partition']
	del compdf['qos']
	del compdf['reqNodes']
	del compdf['reqWalltime']
	del compdf['obsWalltime']
	compdf['obsWaitTime']=compdf['obsWaitTime'].astype(float)
	#observed waittime is label
	compdf.rename(columns={'obsWaitTime':'obsWaitTime_label'},inplace=True)
	compdf.sort_values(by=['jobId'],inplace=True)
	
	#merge the frames on jobid:
	alldf=pd.merge(compdf,queuedf,how='inner',on='jobId')
	del alldf['machine']
	
	#only take values where partition is specified
	alldf.dropna(axis=0,how='any',inplace=True)
	
	alldf.reset_index(drop=True,inplace=True)

	if one_hot:
	#generate class labels for one-hot encoding:
		partitions=list(set(alldf['partition_tag']))
		qosclasses=list(set(alldf['qos_tag']))
		hotdf=alldf.copy()
		hotdf['partition_tag']=alldf.apply(lambda x: partitions.index(x['partition_tag']),axis=1)
		hotdf['qos_tag']=hotdf.apply(lambda x: qosclasses.index(x['qos_tag']),axis=1)
		
		#all columns with "tag" suffixes get one-hot encoded:
		onehotcolumns=[x for x in hotdf.columns[2:] if 'tag' in x]
		for feature in onehotcolumns:
			#what is the number of categories:
			num_cat=np.max(hotdf[feature])+1
			#create dataframe to fill
			fname=feature.split('tag')[0]
			hotcols=[fname+str(c) for c in range(num_cat)]
			tmpcols=['jobId']+hotcols
			tmpdf=pd.DataFrame(columns=tmpcols)
			tmpdf[['jobId']]=hotdf[['jobId']].copy()
			for c in hotcols:
				tmpdf[hotcols]=0.
			#join back the frames
			hotdf=pd.merge(hotdf,tmpdf,how='inner',on='jobId').copy()
			
			#set the hotcols to the correct values
			for i in range(num_cat):
				hotdf.loc[hotdf[feature]==i,fname+str(i)]=1.
	else:
		#just copy as a view
		hotdf=alldf
	
	#delete tag columns
	del hotdf['qos_tag']
	del hotdf['partition_tag']

	return hotdf

def loadQueuedJobData(machine,data_dir,timestamps):
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
		snapFileName = data_dir + "snapshot." + time 
		prioFileName = data_dir + "priority-factors." + time 
	
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


def loadCompletedJobData(machine,start,end,db,user,passwd,hostname,port):
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
	                 #summary.c.completion<=end,
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

if __name__ == '__main__':
    dir = "/global/homes/a/ankitb/qpredict/data/gz-mar21-apr7-16/"
    machine = 'cori'
    tstart = '1459513201'
    tend = '1459910402'
    one_hot = True
    
    queue,completed = get_data(machine,dir,tstart,tend)
    hotdf = create_df(queue,completed,one_hot)

	#print len(coriDebugQ.queuedJobs)
	#print len(coriRegQ.queuedJobs)
	#print len(coriSharedQ.queuedJobs)

	#for k,v in tmplist.items():
	#    print v.getPartition()

	#for k,v in coriJobs.items():
	#	if v.partition == 'shared':
	#		print v.machine,v.jobId,v.partition,v.qos,v.reqWalltime,v.reqNodes,v.priority,v.age,v.fairshare,v.qos_int,v.rank_p
else:
	print "data module loaded"
