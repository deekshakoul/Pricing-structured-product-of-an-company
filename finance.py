
# coding: utf-8

# In[1]:

import csv, numpy as np,matplotlib.pyplot as plt

#Files containing historical data of underlying shares
filename1 = 'lha.csv'
filename2 = 'pah3.csv'
filename3 = 'tka.csv'

col = 4

reader = csv.reader(open(filename1,newline=''))
header = next(reader)
price1 = []
for row in reader:
    price1.append(float(row[col]))
#print(price)
print(np.std(price1)**2)
returns1 = np.ediff1d(price1)/price1[:-1]#Returns of first share
var1 = np.std(returns1)**2 
share1, = plt.plot(price1) 



reader = csv.reader(open(filename2,newline=''))
header = next(reader)
price2 = []
for row in reader:
    price2.append(float(row[col]))
#print(price)
print(np.std(price2)**2)
returns2 = np.ediff1d(price2)/price2[:-1] #Returns for second share
var2 = np.std(returns2)**2
share2, = plt.plot(price2)

reader = csv.reader(open(filename3,newline=''))
header = next(reader)
price3 = []
for row in reader:
    price3.append(float(row[col]))
#print(price)
print(np.std(price3)**2)
returns3 = np.ediff1d(price3)/price3[:-1] #Returns for third share
var3 = np.std(returns3)**2
share3, =plt.plot(price3)


plt.legend([share1,share2,share3],['Lufthansa AG','Porche Automobil Holding SE','ThyssenKrupp AG'])
plt.axis([0,800,0,140 ])

plt.show()


# In[7]:

#print(np.cov(price1,price2))
#print(np.cov(price2,price3))
#print(np.cov(price1,price3))

cov12 = np.cov(returns1,returns2)[0,1]
cov23 = np.cov(returns2,returns3)[0,1]
cov13 = np.cov(returns1,returns3)[0,1]


#Covariance Matrix of Underlying stocks
cov = np.array([[var1,cov12,cov13],[cov12,var2,cov23],[cov13,cov23,var3]])#on daily basis
vol = np.array([np.sqrt(var1), np.sqrt(var2),np.sqrt(var3)])

print('Covariance\n',cov)
print('Volatality of stock1: ',np.sqrt(var1),'\nVolatality of stock2: ',np.sqrt(var2),'\nVolataity of stock3: ',np.sqrt(var3))


# In[4]:

rf = -0.754/100##Drift rate AAAratedBonds
#rf = -0.448/100
dt = 1#In days
sims = 100000 #Number of simulations
steps = 365
bonus = 110/100 

X = np.random.multivariate_normal(
        rf/365 * dt - 0.5 * vol ** 2, cov, (sims,steps));

#Relative prices of the three stocks
price_relative1 = np.exp(np.cumsum(X,axis=1))
price_relative2 = np.exp(np.cumsum(2 * (rf/365 * dt - 0.5 * vol ** 2) - X,axis=1))
dime = price_relative1.shape
#print(price_relative.shape)

K = 0.65#Barrier

#Indices where the barrier is crossed
index1 = np.argmax(price_relative1  < K,axis=1)
index2 = np.argmax(price_relative2  < K,axis=1)
#print(price_relative)
#print(index)
i,k = np.meshgrid(np.arange(dime[0]),np.arange(dime[2]))
#print(i,k)
#print(np.where((price_relative[i.transpose(),index,k.transpose()] < K).any(axis=1),False,True))

#Count of siulations where barrier is not crossed
count1 = np.sum(np.where((price_relative1[i.transpose(),index1,k.transpose()] < K).any(axis=1),False,True))
count2 = np.sum(np.where((price_relative2[i.transpose(),index2,k.transpose()] < K).any(axis=1),False,True))

pay1 = bonus * 1000 #payoff when barrier is not crossed
pay2 = 950#payoff when barrier is crossed
total = count1+count2

#Average Payoff for all simulations
payoff  = (total)/(2*sims) * pay1 + (2*sims - total)/(2*sims) * pay2

print('Payoff after maturity:',payoff)


# In[ ]:



