import csv
import os

dic = {'0':'batch_size','1':'epochs','2':'h_flip','3':'learning_rate_init','4':'lr_decay','5':'momentum','6':'patience','7':'resize_crop','8':'shuffle','9':'tolerance','10':'v_flip','11':'weight_decay'}
with open('importance-single-12params.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')#,quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['dataset', 'parameter', 'marginal_contribution_accuracy', 'marginal_contribution_runtime'])
    d = os.listdir('imps/')
    for i in d:
        #print(i)
        f = open('imps/'+i, 'r')
        lines = f.readlines()
        for line in range(12):
            words = lines[line].split()
            #print(words[0][2:4])
            words_t = lines[line+12].split()
            res = [i.split('-')[1][:-4]]
            if words[0][3] == ',':
                res.append(dic[words[0][2]])
            else:
                res.append(dic[words[0][2:4]])
            res.append(words[3][:-1])
            res.append(words_t[3][:-1])
            writer.writerow(res)