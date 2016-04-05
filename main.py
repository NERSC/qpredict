import os
import sys

import data
import json
from queues import *

import numpy as np
import pandas as pd

def main():
    """Driver routine"""
    # Global params
    dir = "/global/homes/t/tkurth/JOBWAITPREDICT/data/"
    
    with open('/global/homes/t/tkurth/JOBWAITPREDICT/general/mysql_staffdb_creds.json') as mysql_creds:
    	conn_config = json.load(mysql_creds)
    
    hostname=conn_config['host']
    databasename=conn_config['database']
    username=conn_config['user']
    password=conn_config['password']
    portnumber=conn_config['port']
    
    tstart = "1458591001"
    one_hot=True
    
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
        completed = data.loadCompletedJobData(machine,tstart,databasename,username,password,hostname,portnumber)
        
        # I like chinese bears, so I pack the data into a pandas dataframe
        queuedf=pd.DataFrame([x.to_dict() for x in queue.values()])
        #rename columns with class labels to apply one-hot later
        queuedf.rename(columns={'partition':'partition_tag','qos':'qos_tag'},inplace=True)
        queuedf.sort(['jobId'],inplace=True)

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
        compdf.sort(['jobId'],inplace=True)
        
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

        
        #split the sample in training, validation and testing
        train_fraction=0.6
        validation_fraction=0.2
        test_fraction=0.2
        numrows=hotdf.shape[0]
        rows=np.arange(numrows)
        np.random.shuffle(rows)
        shuffledf=hotdf[hotdf.columns[1:]].iloc[rows].reset_index(drop=True)

        #training set
        train_stop=int(np.floor(train_fraction*numrows))
        shuffledf[:train_stop].to_csv('cori_data_train.csv',header=True)
        #validation set
        validation_stop=int(np.floor(validation_fraction*numrows))+train_stop
        shuffledf[train_stop:validation_stop].to_csv('cori_data_validate.csv',header=True)
        #test set:
        shuffledf[validation_stop:].to_csv('cori_data_test.csv',header=True)
        

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
