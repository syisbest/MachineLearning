```python
# -*- coding: utf-8 -*-

"""
Created on Mon Apr 27 17:01:50 2020

@author: 沈岩
"""

import csv
import random

#读数据
with open('C:/Users/沈岩/Desktop/breast-cancer.csv','r') as file:
    reader=csv.DictReader(file)
    datas=[row for row in reader]

#把数据打乱
random.shuffle(datas)
#取1/3数据来测试，2/3数据来训练    
n=len(datas)//3
#测试集和训练集
test_set=datas[0:n]
train_set=datas[n:]
#求距离（欧式）
def distance(d1,d2):
    #结果
    res=0
    #多维，即(x1-x0)^2+(y1-y0)^2+(z1-z0)^2+...     
    for key in("radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean",
               "concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se",	
               "smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
               "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst",
               "concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"):
        res+=(float(d1[key])-float(d2[key]))**2
    #取平方根
    return res**0.5
#设置K值
K=5
#KNN
def KNN(data):
    #存距离的结果
    res=[
            #存储成字典
            {"result":train['diagnosis'],"distance":distance(data,train)}
            #从训练集里取数据
            for train in train_set
            ]
    #结果排序，按distance排序
    res=sorted(res,key=lambda item:item["distance"])
    #分片，只需要前K个数据
    res2=res[0:K]
    #存权重
    result={'B':0,'M':0}
    #所有距离之和
    sum=0
    #遍历求和
    for r in res2:
        sum+=r['distance']
    #求结果占比
    for r in res2:
        result[r['result']]+=1-r['distance']/sum
    #谁占比大返回谁
    if result['B']>result['M']:
        return 'B'
    else:
        return 'M'

print("1707004642沈岩")
#求第一个数据的结果
print("第一位病人的预测结果是"+KNN(test_set[0]))

#测试正确的个数
correct=0
#从测试集里取数据
for test in test_set:
    #拿到正确的结果
    result=test['diagnosis']
    #计算由KNN得到的结果
    result2=KNN(test)
    #结果一致就加一
    if result==result2:
        correct+=1
#正确的结果比上所有的结果    
print("准确率{:.2f}%".format(100*correct/len(test_set)))
```

------

**结果：**

![](C:\Users\沈岩\Desktop\1.png)

**数据集来源：https://www.kaggle.com/yuqing01/breast-cancer**

