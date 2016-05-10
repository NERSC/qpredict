import os
import sys
import shutil

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
from datetime import datetime as dt
from queues import *

#this is canonical now
labelname='obsWaitTime_label'
featurelist=['age','fairshare','priority','qos_int','rank_p','reqNodes','reqWalltime','timeOfDay','workAhead']
onehotfeaturelist=['partition','qos','weekday']
weekdays=['monday','tuesday','wednesday','thursday','friday','saturday','sunday']

#small helper function to get the time of day
def epoch_to_timeofday(epoch):
	ts=dt.fromtimestamp(epoch)
	return (ts - ts.replace(hour=0,minute=0,second=0)).total_seconds()


def get_data(machine,base_dir,temp_dir,data_dir,db_cred_file,tstart,tend):
	"""	Creates a new test set independent from one used in training/validation/test sequence """
	
	snapshot_file = base_dir+"input/snapshots_"+machine+".txt"

	print 'Determining snapshots'
	#get all files
	allfiles=set(os.listdir(data_dir))
	#get relevant files and filter by start end end times
	snapfiles=[x for x in allfiles if 'snapshot' in x]
	snaptimes={int(x.split('.')[1]) for x in snapfiles}
	priofiles=[x for x in allfiles if 'priority-factors' in x]
	priotimes={int(x.split('.')[1]) for x in priofiles}
	commontimes=snaptimes.intersection(priotimes)
	#get all timestanmps between bounds
	timestamps=sorted(list({str(x) for x in commontimes if x>=int(tstart) and x<=int(tend)}))
	print 'Done'

	# For completed jobs database
	with open(db_cred_file) as mysql_creds:
		conn_config = json.load(mysql_creds)

	hostname=conn_config['host']
	databasename=conn_config['database']
	username=conn_config['user']
	password=conn_config['password']
	portnumber=conn_config['port']

	queue = {}
	completed = {}

	print 'Loading queue data'
	# Get current queue data
	queue = data.loadQueuedJobData(machine,data_dir,temp_dir,timestamps)
	print 'Done'
	
	print 'Loading completed jobs data'
	# Get completed jobs data
	completed = data.loadCompletedJobData(machine,tstart,tend,databasename,username,password,hostname,portnumber)	
	print 'Done'

	return queue, completed


#do one-hot encoding on columns tagged by <name>_tag
def one_hot_encode(inputdf):
	#copy input:
	hotdf=inputdf.copy()
	
	#all columns with "tag" suffixes get one-hot encoded:
	onehotcolumns=[x for x in hotdf.columns[:] if 'tag' in x]

	#one-hot-encode one-by-one
	for feature in onehotcolumns:
		#all possible values the feature can acquire, sort the list to make sure that it is the same
		#for different calls to this function
		categorylist=sorted(list(set(inputdf[feature])))
		
		#print the categories, useful for seeing which one-hot encoded features map to which tag:
		print "Categories for feature ",feature,": ",categorylist
		
		#assign numerical value
		hotdf[feature]=inputdf.apply(lambda x: categorylist.index(x[feature]),axis=1)
		
		#what is the number of categories:
		num_cat=len(categorylist)
		#create empty dataframe to fill
		fname=feature.split('tag')[0]
		hotcols=[fname+str(c) for c in range(num_cat)]
		tmpdf=pd.DataFrame(index=np.arange(hotdf.shape[0]),columns=hotcols)
		for c in hotcols:
			tmpdf[hotcols]=0.
		#join back the frames. merge on index, do not create another ID field
		hotdf=pd.merge(hotdf,tmpdf,how='inner',left_index=True,right_index=True).copy()
		
		#set the hotcols to the correct values
		for i in range(num_cat):
			hotdf.loc[hotdf[feature]==i,fname+str(i)]=1.
		
		#remove feature as it is one-hot encoded now
		del hotdf[feature]
	
	#return result
	return hotdf


#create (one-hot-encoded) dataframe from json input:
def create_df_from_json(inputjson, one_hot):
	#convert to df:
	inputdf=pd.read_json(inputjson)
	inputdf=inputdf.reindex_axis(featurelist+[z+'_tag' for z in onehotfeaturelist],axis=1)
	
	#rename one-hot columns
	ohrenamedict={x[0]:x[1] for x in zip(onehotfeaturelist,[z+'_tag' for z in onehotfeaturelist])}
	inputdf.rename(columns=ohrenamedict,inplace=True)
	
	#reset-index
	inputdf.reset_index(drop=True,inplace=True)

	#one-hot-encode:
	if one_hot:
		return one_hot_encode(inputdf)
	else:
		for feature in onehotfeaturelist:
			del inputdf[feature]
		return inputdf


def create_df(queue, completed, one_hot):
	"""Packs queued and completed job data in dataframes"""
	#Pack it in dataframes
	queuedf=pd.DataFrame([x.to_dict() for x in queue.values()])
	queuedf.dropna(axis=0,inplace=True)

	#precompute some features and get rid of the eligibleTime
	queuedf=queuedf[queuedf.eligibleTime>0]
	queuedf['weekday'] = queuedf.apply(lambda x: dt.fromtimestamp(float(x['eligibleTime'])).weekday(),axis=1)
	queuedf['timeOfDay'] = queuedf.apply(lambda x: epoch_to_timeofday(float(x['eligibleTime'])),axis=1)
	queuedf['obsWaitTime'] = queuedf['startTime'] - queuedf['eligibleTime']
	del queuedf['eligibleTime']
	del queuedf['startTime']
	
	#rename columns with class labels to apply one-hot later
	ohrenamedict={x[0]:x[1] for x in zip(onehotfeaturelist,[z+'_tag' for z in onehotfeaturelist])}
	queuedf.rename(columns=ohrenamedict,inplace=True)
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
	compdf.rename(columns={'obsWaitTime':labelname},inplace=True)
	compdf.sort_values(by=['jobId'],inplace=True)
	
	#merge the frames on jobid:
	alldf=pd.merge(compdf,queuedf,how='inner',on='jobId')
	del alldf['machine']
	
	#only take values where partition is specified
	alldf.dropna(axis=0,how='any',inplace=True)

	#reorder to globally common order:
	alldf=alldf.reindex_axis(['jobId',labelname]+featurelist+[z+'_tag' for z in onehotfeaturelist],axis=1)
	
	#reset index after shuffling x and y.
	alldf.reset_index(drop=True,inplace=True)
	
	#determine if one-hot encoding is necessary
	if one_hot:
		hotdf=one_hot_encode(alldf)
	else:
		#just copy as a view
		hotdf=alldf
		for feature in onehotfeaturelist:
			del hotdf[feature]

	print hotdf
	print hotdf.columns

	return hotdf


def loadQueuedJobData(machine,data_dir,temp_dir,timestamps):
	"""Loads jobs data from stored snapshots. One subtlety here 
	is that the sprio snapshot needs to be read first since it only 
	stores jobs that are actually in queued state. The sqs output 
	also contains jobs that are in states other than queued and those
	need to be excluded for our purpose. The returned object contains 
	a list of unique jobs in the queue from a given interval. 		
	"""
	# Temp dict object
	tempJobs = {}
	
	#csv file which is read or written:
	csvfile='queuedata_'+str(timestamps[0])+'_to_'+str(timestamps[-1])+'.csv'
	
	#if os.path.isfile(data_dir + csvfile):
		#aggregatedf.read_csv(data_dir + csvfile)
	
	# used for creating a comprehensive DF
	datacols=['jobId','timestamp',
				'priority', 'age', 'fairshare',
				'qos_int', 'rank_p', 'partition',
				'qos','reqWalltime','reqNodes','eligibleTime']
	#aggregatedf=pd.DataFrame(columns=datacols)
	
	#loop over files
	for time in timestamps:
		snapFileName = "snapshot." + time 
		prioFileName = "priority-factors." + time 
	
		#check if files are unzipped already, if not, unzip
		if not os.path.isfile(temp_dir + snapFileName):
			#copy file to temp-directory:
			try:
				shutil.copy (data_dir + snapFileName + ".gz", temp_dir + snapFileName + ".gz")
				os.system("gunzip" + " " + temp_dir + snapFileName + ".gz")
			except:
				print "File ",data_dir + snapFileName + ".gz", " not found, continue!"
				continue
			
		if not os.path.isfile(temp_dir + prioFileName):
			#copy to temp-directory
			try:
				shutil.copy (data_dir + prioFileName + ".gz", temp_dir + prioFileName + ".gz")
				os.system("gunzip" + " " + temp_dir + prioFileName + ".gz")
			except:
				print "File ",data_dir + prioFileName + ".gz", " not found, continue!"
				continue
		
		#start reading data
		count = 0
		uniqueJobsInThisSnapshot = []
		allJobsInThisSnapshot = []
		
		# Do stuff with the data
		with open(temp_dir + prioFileName) as pf:
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
					tempJobs[jobId] = QueuedJob(machine,jobId,None,None,None,None,None,priority,age,fairshare,qos_int,rank_p)
					uniqueJobsInThisSnapshot.append(jobId)
					
				#append to dataframe
				allJobsInThisSnapshot.append({'jobId':jobId,'timestamp':time,
												'priority':priority, 'age':age, 
												'fairshare':fairshare, 'qos_int': qos_int,
												'rank_p': rank_p})
		
		#temporary df:
		tmpdf1=pd.DataFrame(allJobsInThisSnapshot)
		if tmpdf1.empty:
			print "Priority file ",temp_dir + prioFileName," is empty, skipping!"
			continue
		
		#second set
		allJobsInThisSnapshot = []
		
		with open(temp_dir + snapFileName) as sf:
			sf.readline()	
			for line in sf.readlines():
				jobs = line.split()
				jobId = int(jobs[0])
				eligt = jobs[10]
				if 'N/A' in eligt:
					eligt=None
				else:
					eligt=int(eligt)
				
				#push job into list:
				allJobsInThisSnapshot.append({'jobId': jobId, 
											'partition': jobs[3],
											'qos': jobs[4].split('_')[0],
											'reqNodes': int(jobs[5]),
											'reqWalltime': aux.convertWalltimeToSecs(jobs[7]),
											'eligibleTime': eligt
											})
				
				if jobId in uniqueJobsInThisSnapshot:
					#if eligible, everything is golden
					if eligt:
						#if eligible, everything is golden
						tempJobs[jobId].partition = jobs[3]
						tempJobs[jobId].qos = jobs[4].split('_')[0]
						tempJobs[jobId].reqNodes = int(jobs[5])	
						tempJobs[jobId].reqWalltime = aux.convertWalltimeToSecs(jobs[7])
						tempJobs[jobId].priority = int(jobs[8])
						tempJobs[jobId].eligibleTime=eligt
					else:
						#remove the job, might be added back in later on
						del tempJobs[jobId]
		
		
		#append the temporary dataframe to the aggregate one:
		tmpdf2=pd.DataFrame(allJobsInThisSnapshot)
		if tmpdf2.empty:
			print "Snapshot file ",temp_dir + snapFileName," is empty, skipping!"
			continue
		
		#merge
		tmpdf1=pd.merge(tmpdf1,tmpdf2,on='jobId')
		
		#sort by rank and drop nan:
		tmpdf1.sort_values(by=['rank_p'],inplace=True)
		tmpdf1.dropna(axis=0,inplace=True)
		tmpdf1.reset_index(drop=True,inplace=True)
		
		#compute cumulative sum for workload-ahead:
		tmpdf1['workAhead']=tmpdf1.apply(lambda x: x['reqWalltime']*x['reqNodes']*10**(-6),axis=1).cumsum().reset_index(drop=True)
		#tmpdf1['workAheadStd']=tmpdf1.apply(lambda x: x['reqWalltime']*x['reqNodes']*10**(-6),axis=1).cummean().reset_index(drop=True)
		tmpdf1['workAhead']=tmpdf1['workAhead'].shift()
		tmpdf1.fillna(0.,axis=0,inplace=True)
		tmpdf1.set_index('jobId',inplace=True)
		
		#add the feature to the jobs in joblist:
		for job in set(tmpdf1.index):
			if job in tempJobs:
				if not tempJobs[job].workAhead:
					tempJobs[job].workAhead=tmpdf1.get_value(job,'workAhead')
					#tempJobs[job].workAheadStd=tmpdf1.get_value(job,'workAheadStd')
		
		#Concat to aggregatedf
		#aggregatedf=pd.concat([aggregatedf,tmpdf1])
		
		# delete the copied files
		if os.path.isfile(temp_dir + snapFileName + ".gz"):
			os.remove(temp_dir + snapFileName + ".gz")
		if os.path.isfile(temp_dir + snapFileName):
			os.remove(temp_dir + snapFileName)
		if os.path.isfile(temp_dir + prioFileName + ".gz"):
			os.remove(temp_dir + prioFileName + ".gz")
		if os.path.isfile(temp_dir + prioFileName):
			os.remove(temp_dir + prioFileName)

	#store aggregate DF
	#aggregatedf.reset_index(drop=True,inplace=True)
	#aggregatedf.to_csv(data_dir + 'queuedata_'+str(timestamps[0])+'_to_'+str(timestamps[-1])+'.csv',index=False)
	
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
		#ATTENTION, the wait secs stored in the DB does not account for ineligibility
		obsStartTime = data[4]
									
		finishedJobs[jobId] = CompletedJob(machine,jobId,partition,qos,reqWalltime,reqNodes,obsWalltime,obsStartTime)
							
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
