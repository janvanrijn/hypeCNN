import csv
import math

with open('stl10-responses-time.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')#,quotechar='"', quoting=csv.QUOTE_MINIMAL)
	f = open('output-all.txt', 'r')
	lines = f.readlines()
	for line in lines:
		words = line.split()
		res = [words[3]]
		# res = []
		# res.append(int(math.log(int(words[6][:-1]),2))-3)
		# res.append(int(words[4])+1)
		# res.append(words[10][:-1])
		# res.append(words[12][:-1])
		# res.append(words[14][:-1])
		writer.writerow(res)
