import matplotlib.pyplot as plt
import numpy as np

def data_to_array(path):
    x=[]
    y=[]
    d=[]
    data=open(path,'+r')
    tmp=data.readlines()
    for f in tmp:
        lis=f.split(',')
        x.append(int(lis[0]))
        y.append(int(lis[1]))
        d.append(int(lis[2]))
    x=np.array(x)
    y=np.array(y)
    d=np.array(d)
    return x,y,d
color=['c', 'b', 'g', 'r', 'm', 'gold', 'k', 'lime','pink']
path='./calib_data/'
plt.figure(figsize=(8,6))
for i in range(1,10):
    datapath=path+str(i)+'.txt'
    x,y,d=data_to_array(datapath)
    plt.scatter(x,y,label=str(i),marker='.',c=color[i-1])

plt.legend()
plt.show()