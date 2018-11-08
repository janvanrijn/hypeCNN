import csv
import math

with open('cifar10-features.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')#,quotechar='"', quoting=csv.QUOTE_MINIMAL)
	f = open('output-all.txt', 'r')
	lines = f.readlines()
	for line in lines:
		words = line.split()
		#res = [words[1]]
		res = []
		res.append(int(math.log(int(words[5][:-1]),2))-5)
		res.append(words[7][:-1])
		res.append(words[9][:-1])
		res.append(words[11][:-1])
		res.append(words[13][:-1])
		writer.writerow(res)
