def convertWalltimeToSecs(str):
	if str.count(':') == 0:
		secs = int(str)
	elif str.count(':') == 1:
		res = str.split(':')
		secs = 60*int(res[0]) + int(res[1])
	elif str.count(':') == 2:
		if(str.count('-') == 0):
			res = str.split(':')
			secs = 3600*int(res[0]) + 60*int(res[1]) + int(res[2])
		else:
			d_str = str.split('-')
			secs = 86400*int(d_str[0])
			
			res = d_str[1].split(':')
			secs += 3600*int(res[0]) + 60*int(res[1]) + int(res[2])

	#print secs

	return secs
