class BatchQueue:
	def __init__(self,machine):	
		self.machine = machine
		self.wallLimit = 0  # in secs
		self.nodeLimit = 0
		self.queuedJobs = {} # Dict of jobs indexed by job id

	def addJob(self,jobId,partition,qos,walltime,nodes,priority,rank_p):
		self.queuedJobs[jobId] = Job(jobId,partition,qos,walltime,nodes,priority,rank_p)

class DebugQueue(BatchQueue):
	def __init__(self,machine):
		BatchQueue.__init__(self,machine)
		if machine == 'cori':
			self.wallLimit = 1800
			self.nodeLimit = 112
		elif machine == 'edison':
			self.wallLimit = 1800
			self.nodeLimit = 512

		self.qos = 'debug'

class RegQueue(BatchQueue):
    def __init__(self,machine):
        BatchQueue.__init__(self,machine)
        if machine == 'cori':
            BatchQueue.wallLimit = 172800
            BatchQueue.nodeLimit = 1420
        elif machine == 'edison':
            BatchQueue.wallLimit = 129600
            BatchQueue.nodeLimit = 5462

	self.qos = ['scavenger', 'low', 'normal', 'premium']

class SharedQueue(BatchQueue):
    def __init__(self,machine):
        BatchQueue.__init__(self,machine)
        if machine == 'cori':
            BatchQueue.wallLimit = 172800
            BatchQueue.nodeLimit = 16
        elif machine == 'edison':
            BatchQueue.wallLimit = None
            BatchQueue.nodeLimit = None

	self.qos = ['scavenger', 'low', 'normal', 'premium']

class Job:
	def __init__(self,machine,jobId,partition,qos,reqWalltime,reqNodes):
		# From snapshots
		self.jobId = jobId
		self.machine = machine
		self.partition = partition
		self.qos = qos
		self.reqWalltime = reqWalltime
		self.reqNodes = reqNodes

		def getPartition(self):
			return self.partition


class QueuedJob(Job):
	def __init__(self,machine,jobId,partition,qos,reqWalltime,reqNodes,priority,age,fairshare,qos_int,rank_p):
		Job.__init__(self,machine,jobId,partition,qos,reqWalltime,reqNodes)
		# Known
		self.priority = priority
		self.age = age
		self.fairshare = fairshare
		self.qos_int = qos_int
		self.rank_p = rank_p

	# TBD
	predictedWaitTime = 0

	def getPartition(self):
		return self.partition

	def predictWaitTime(self):
		#TODO this is where output is stored
		self.predictedWaitTime = 0

class CompletedJob(Job):
	def __init__(self,machine,jobId,partition,qos,reqWalltime,reqNodes,obsWalltime,obsWaitTime):
		Job.__init__(self,machine,jobId,partition,qos,reqWalltime,reqNodes)
		self.obsWaitTime = obsWaitTime
		self.obsWalltime = obsWalltime

# Debug/test
if __name__ == "__main__":
	edison_dbg = DebugQueue("edison")
	print edison_dbg.nodeLimit

	cori_dbg = DebugQueue("cori")
	print cori_dbg.wallLimit
	print cori_dbg.qos
else:
	print(__name__ + " module loaded") 
