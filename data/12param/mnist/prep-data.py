import csv
import math

with open('mnist-features.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')#,quotechar='"', quoting=csv.QUOTE_MINIMAL)
	f = open('output-all.txt', 'r')
	lines = f.readlines()
	for line in lines:
		words = line.split()
		#res = [words[3]]
		res = []
		res.append(int(math.log(int(words[6][:-1]),2))-5)
		res.append(int(words[4])+1)
		if words[10][:-1]=='False':
			res.append(1)
		else:
			res.append(0)
		res.append(words[12][:-1])
		res.append(words[14][:-1])
		res.append(words[16][:-1])
		res.append(words[18][:-1])
		if words[20][:-1]=='False':
			res.append(1)
		else:
			res.append(0)
		if words[22][:-1]=='False':
			res.append(1)
		else:
			res.append(0)
		res.append(words[24][:-1])
		if words[26][:-1]=='False':
			res.append(1)
		else:
			res.append(0)
		res.append(words[28][:-1])
		writer.writerow(res)
