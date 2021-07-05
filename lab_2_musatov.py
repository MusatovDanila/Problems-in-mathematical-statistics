import matplotlib.pyplot as plt
import numpy as np
import math
import plotly.graph_objects as go
import copy
import xlsxwriter

N=200
v=67
mu=((-1)**v)*0.01*v #A
L=2+v*(0.01)*((-1)**v) #lambda
s_d=1+(0.01*v) #standard deviation
a=((-1)**v)*0.05*v
b=a+6

#beta=1/L
#Normal=np.random.normal(mu, s_d, N)
#Indicative=np.random.exponential(beta, N) 
#uniform=np.random.uniform(a,b, N)

#np.savetxt('normal.txt', Normal,'%f')
#np.savetxt('indicative.txt', Indicative,'%f')
#np.savetxt('uniform.txt', uniform,'%f')

my_file = open("normal.txt", "r")
Normal=[]
for num in my_file:
    Normal.append(float(num))
my_file.close()
Normal_unsort = []
Normal_unsort = copy.deepcopy(Normal)
my_file1 = open("indicative.txt", "r")
Indicative=[]
for num in my_file1:
    Indicative.append(float(num))
my_file1.close()
Indicative_unsort = []
Indicative_unsort = copy.deepcopy(Indicative)
my_file2 = open("uniform.txt", "r")
uniform=[]
for num in my_file2:
    uniform.append(float(num))
my_file2.close()
uniform_unsort = []
uniform_unsort = copy.deepcopy(uniform)

Normal.sort()
Indicative.sort()
uniform.sort()

m=1+int(math.log(N,2))
n_0=min(Normal)
n_m=max(Normal)
d_n=n_m-n_0
i_0=0
i_m=max(Indicative)
d_i=i_m-i_0
d_u=b-a

a_n=[n_0]
for i in range(m-1):
    a_n.append(a_n[i]+d_n/m)
a_n.append(n_m)

a_i=[i_0]
for i in range(m-1):
    a_i.append(a_i[i]+d_i/m)
a_i.append(i_m)

a_u=[a]
for i in range(m-1):
    a_u.append(a_u[i]+d_u/m)
a_u.append(b)

n_n=[]
counter = 0
k=1
helper = a_n[k]
for i in range(len(Normal)):
    if Normal[i]<=helper:
        counter += 1 
    else:
        n_n.append(counter)
        k += 1
        helper = a_n[k]
        if (Normal[i]<=helper):
            counter=1
        else: counter=0
n_n.append(counter)

n_i=[]
counter = 0
k=1
helper = a_i[k]
for i in range(len(Indicative)):
    if Indicative[i]<=helper:
        counter += 1 
    else:
        n_i.append(counter)
        k += 1
        helper = a_i[k]
        if (Indicative[i]<=helper):
            counter=1
        else:
            counter=0
n_i.append(counter)

n_u=[]
counter = 0
k=1
helper = a_u[k]
for i in range(len(uniform)):
    if uniform[i]<=helper:
        counter += 1 
    else:
        n_u.append(counter)
        k += 1
        helper = a_u[k]
        if (uniform[i]<=helper):
            counter=1
        else: counter=0
n_u.append(counter)
w_n=[]
for i in range(len(n_n)):
    w_n.append(n_n[i]/N)
    
w_i=[]
for i in range(len(n_i)):
    w_i.append(n_i[i]/N)
    
w_u=[]
for i in range(len(n_u)):
    w_u.append(n_u[i]/N)

x_n = []
for i in range(len(a_n)-1):
    x_n.append((a_n[i]+a_n[i+1])/2)
    
x_i = []
for i in range(len(a_i)-1):
    x_i.append((a_i[i]+a_i[i+1])/2)
    
x_u = []
for i in range(len(a_u)-1):
    x_u.append((a_u[i]+a_u[i+1])/2)

math_exp_n=0
for i in range(len(x_n)):
    math_exp_n+=(x_n[i]*w_n[i])
print(math_exp_n)

math_exp_i=0
for i in range(len(x_i)):
    math_exp_i+=(x_i[i]*w_i[i])
print(math_exp_i)

math_exp_u=0
for i in range(len(x_u)):
    math_exp_u+=(x_u[i]*w_u[i])
print(math_exp_u)

h_n=(n_m-n_0)/m
h_i=(i_m-i_0)/m
h_u=(b-a)/m

Disp_n=0
for i in range(len(x_n)):
    Disp_n+=w_n[i]*((x_n[i]-math_exp_n)**2)
Disp_n-=(h_n**2)/12
print(Disp_n)

Disp_i=0
for i in range(len(x_i)):
    Disp_i+=w_i[i]*((x_i[i]-math_exp_i)**2)
Disp_i-=(h_i**2)/12
print(Disp_i)

Disp_u=0
for i in range(len(x_u)):
    Disp_u+=w_u[i]*((x_u[i]-math_exp_u)**2)
Disp_u-=(h_u**2)/12
print(Disp_u)

stand_dev_n=math.sqrt(Disp_n)
stand_dev_i=math.sqrt(Disp_i)
stand_dev_u=math.sqrt(Disp_u)

k=n_n.index(max(n_n))
Moda_n=a_n[k]+(h_n*(w_n[k]-w_n[k-1])/(2*w_n[k]-w_n[k-1]-w_n[k+1]))
s_n=0
i=0
Med_n=0
while s_n<0.5:
    s_n+=w_n[i]
    i+=1
if s_n==0.5:
    Med_n=a_n[i]
else:
    s_n-=w_n[i-1]
    Med_n=a_n[i-1]+(h_n/w_n[i-1])*(0.5-s_n)
print(Moda_n)
print(Med_n,'\n')

k=n_i.index(max(n_i))
Moda_i=a_i[k]+(h_i*(w_i[k])/(2*w_i[k]-w_i[k+1]))
s_i=0
i=0
Med_i=0
while s_i<0.5:
    s_i+=w_i[i]
    i+=1
if s_i==0.5:
    Med_i=a_i[i]
else:
    s_i-=w_i[i-1]
    Med_i=a_i[i-1]+(h_i/w_i[i-1])*(0.5-s_i)
print(Moda_i)
print(Med_i,'\n')

k=n_u.index(max(n_u))
Moda_u=a_u[k]+(h_u*(w_u[k]-w_u[k-1])/(2*w_u[k]-w_u[k-1]-w_u[k+1]))
s_u=0
i=0
Med_u=0
while s_u<0.5:
    s_u+=w_u[i]
    i+=1
if s_u==0.5:
    Med_u=a_u[i]
else:
    s_u-=w_u[i-1]
    Med_u=a_u[i-1]+(h_u/w_u[i-1])*(0.5-s_u)
print(Moda_u)
print(Med_u,'\n')
c_m_3_n=0
c_m_4_n=0
for i in range(len(x_n)):
    c_m_3_n+=w_n[i]*((x_n[i]-math_exp_n)**3)
    c_m_4_n+=w_n[i]*((x_n[i]-math_exp_n)**4)
c_as_n=c_m_3_n/(stand_dev_n**3)
c_ex_n= (c_m_4_n/(stand_dev_n**4)-3)

c_m_3_i=0
c_m_4_i=0

for i in range(len(x_i)):
    c_m_3_i+=w_i[i]*((x_i[i]-math_exp_i)**3)
    c_m_4_i+=w_i[i]*((x_i[i]-math_exp_i)**4)

c_as_i=c_m_3_i/(stand_dev_i**3)
c_ex_i= ((c_m_4_i/(stand_dev_i**4))-3)

c_m_3_u=0
c_m_4_u=0

for i in range(len(x_u)):
    c_m_3_u+=w_u[i]*((x_u[i]-math_exp_u)**3)
    c_m_4_u+=w_u[i]*((x_u[i]-math_exp_u)**4)
    
c_as_u=c_m_3_u/(stand_dev_u**3)
c_ex_u= ((c_m_4_u/(stand_dev_u**4))-3)

S=[1/N]
for i in range(1,N):
    S.append(S[i-1]+(1/N))

fig = plt.figure(figsize=(20, 10))
plt.hlines(0, Normal[0]-0.05, Normal[0])
for i in range(len(S)-1):
    plt.hlines(S[i], Normal[i], Normal[i+1])
plt.hlines(S[i+1], Normal[i+1], Normal[i+1]+0.05) 
plt.title('Эмпирическая функция нормального распределения',fontsize =15)
plt.ylabel('s - sum(w(x)) ', fontsize =15)
plt.yticks(np.arange(0.0, 1.05, step=0.05), fontsize =12)
plt.xticks(np.arange(round(Normal[0],1), round(Normal[len(Normal)-1]+0.3,1), step=0.3), fontsize =12)
plt.xlabel('x - value', fontsize =15)
plt.savefig('normal_emp.jpg')

fig = plt.figure(figsize=(20, 10))
plt.hlines(0, -0.05, Indicative[0])
for i in range(len(S)-1):
    plt.hlines(S[i], Indicative[i], Indicative[i+1])
plt.hlines(S[i+1], Indicative[i+1], Indicative[i+1]+0.05)    
plt.title('Эмпирическая функция показательного распределения',fontsize =15)
plt.ylabel('s - sum(w(x)) ', fontsize =15)
plt.yticks(np.arange(0.0, 1.05, step=0.05), fontsize =12)
plt.xticks(np.arange(round(Indicative[0],1), round(Indicative[len(Indicative)-1],1)+0.2, step=0.1), fontsize =12)
plt.xlabel('x - value', fontsize =15)
plt.savefig('indicative_emp.jpg')

fig = plt.figure(figsize=(20, 10))
plt.hlines(0, uniform[0]-0.05, uniform[0])
for i in range(len(S)-1):
    plt.hlines(S[i], uniform[i], uniform[i+1])
plt.hlines(S[i+1], uniform[i+1], uniform[i+1]+0.05)    
plt.title('Эмпирическая функция показательного распределения',fontsize =15)
plt.ylabel('s - sum(w(x)) ', fontsize =15)
plt.yticks(np.arange(0.0, 1.05, step=0.05), fontsize =12)
plt.xticks(np.arange(round(uniform[0],1), round(uniform[len(uniform)-1]+0.2,1), step=0.2), fontsize =12)
plt.xlabel('x - value', fontsize =15)
plt.savefig('uniform_emp.jpg')

y_n=[]
for i in range(len(w_n)):
    y_n.append(w_n[i]/h_n)

fig = plt.figure(figsize=(25, 15))
plt.hist(Normal,bins=m, density=True)
plt.title('Гистограмма нормального распределения',fontsize =15)
plt.yticks(np.arange(0.0, 0.250, step=0.025), fontsize =12)
plt.xticks(np.arange(round(Normal[0]-0.1,1), round(Normal[len(Normal)-1]+0.1,1), step=0.2), fontsize =10)
plt.savefig('normal_hist.jpg')

fig = plt.figure(figsize=(20, 10))
plt.hist(Indicative,bins=m, density=True)
plt.title('Гистограмма показательного распределения',fontsize =15)
plt.yticks(np.arange(0.0, 0.9, step=0.05), fontsize =12)
plt.xticks(np.arange(round(Indicative[0],1), round(Indicative[len(Indicative)-1]+0.1,1), step=0.1), fontsize =12)
plt.savefig('indicative_hist.jpg')

fig = plt.figure(figsize=(20, 10))
plt.hist(uniform,bins=m, density=True)
plt.title('Гистограмма равномерного распределения',fontsize =15)
plt.yticks(np.arange(0.0, 0.275, step=0.025), fontsize =12)
plt.xticks(np.arange(round(uniform[0],1)-0.1, round(uniform[len(uniform)-1]+0.2,1), step=0.2), fontsize =12)
plt.savefig('uniform_hist.jpg')

def normal_prob(miu, stdev, x):
    return 0.5 * (1 + math.erf((x-miu)/(stdev * 2**0.5)))

P_n=[]
for i in range(len(a_n)-1):
    P_n.append(normal_prob(mu,s_d,a_n[i+1])-normal_prob(mu,s_d,a_n[i]))

P_i=[]
for i in range(len(a_i)-1):
    P_i.append(math.exp(-a_i[i]*L)-math.exp(-a_i[i+1]))
P_u=[]
for i in range(len(a_u)-1):
    P_u.append((a_u[i+1]-a_u[i])/(b-a))

Disp_n_t=s_d**2
c_as_i_t=2
c_ex_i_t=6
Med_i_t=math.log(2)/L
s_d_i_t=L**(-1)
Disp_i_t=L**(-2)
math_exp_i_t=L**(-1)
Moda_i_t=0

c_as_u_t=0
c_ex_u_t=-6/5
Moda_u_t=(a+b)/2
Med_u_t=(a+b)/2
s_d_u_t=(b-a)/(2*math.sqrt(3))
Disp_u_t=((b-a)**2)/12
math_exp_u_t=(a+b)/2

a_n_h =[]
w_n_h = []
P_n_h = []
abs_n_w_p_h = []
h="["+str(round(a_n[0],5))+" , "+str(round(a_n[1],5))+"]"
a_n_h.append(h)
for i in range(1,len(a_n)-1):
    a_n_h.append("("+str(round(a_n[i],5))+" , "+str(round(a_n[i+1],5))+"]")
a_n_h.append(' ')
for i in range(len(w_n)):
    w_n_h.append(round(w_n[i], 5))
w_n_h.append(sum(w_n))
for i in range(len(P_n)):
    P_n_h.append(round(P_n[i], 5))
P_n_h.append(round(sum(P_n)))
for i in range(len(w_n)):
    abs_n_w_p_h.append(round((abs(w_n[i]-P_n[i])), 5))
max_n_delt = max(abs_n_w_p_h)
abs_n_w_p_h.append(max_n_delt)

x_n_h =[]
n_n_h =[]
for i in range(len(x_n)):
    x_n_h.append(round(x_n[i],5))
x_n_h.append(' ')
for i in range(len(n_n)):
    n_n_h.append(round(n_n[i], 5))
n_n_h.append(sum(n_n))

fig = go.Figure(data=[go.Table(header=dict(values=["Интервалы", 'n_ j' , 'w_ j']),
cells=dict(values=[a_n_h, n_n_h, w_n_h]))
])
fig.show()

fig = go.Figure(data=[go.Table(header=dict(values=["Интервалы", 'w`_ j' , 'p_ j', '|w`_ j - p_ j|']),
cells=dict(values=[a_n_h, w_n_h, P_n_h, abs_n_w_p_h]))
])
fig.show()

a_i_h =[]
w_i_h = []
P_i_h = []
abs_i_w_p_h = []
a_i_h.append("["+str(round(a_i[0],5))+" , "+str(round(a_i[1],5))+"]")
for i in range(1,len(a_i)-1):
    a_i_h.append("("+str(round(a_i[i],5))+" , "+str(round(a_i[i+1],5))+"]")
a_i_h.append(' ')
for i in range(len(w_i)):
    w_i_h.append(round(w_i[i], 5))
w_i_h.append(sum(w_i))
for i in range(len(P_i)):
    P_i_h.append(round(P_i[i], 5))
P_i_h.append(round(sum(P_i)))
for i in range(len(w_i)):
    abs_i_w_p_h.append(round((abs(w_i[i]-P_i[i])), 5))
max_i_delt = max(abs_i_w_p_h)
abs_i_w_p_h.append(max_i_delt)

x_i_h =[]
n_i_h =[]
for i in range(len(x_i)):
    x_i_h.append(round(x_i[i],5))
x_i_h.append(' ')
for i in range(len(n_i)):
    n_i_h.append(round(n_i[i], 5))
n_i_h.append(sum(n_i))

fig = go.Figure(data=[go.Table(header=dict(values=["Интервалы", 'n_ j' , 'w_ j']),
cells=dict(values=[a_i_h, n_i_h, w_i_h]))
])
fig.show()

fig = go.Figure(data=[go.Table(header=dict(values=["Интервалы", 'w`_ j' , 'p_ j', '|w`_ j - p_ j|']),
cells=dict(values=[a_i_h, w_i_h, P_i_h, abs_i_w_p_h]))
])
fig.show()

a_u_h =[]
w_u_h = []
P_u_h = []
abs_u_w_p_h = []
a_u_h.append("["+str(round(a_u[0],5))+" , "+str(round(a_u[1],5))+"]")
for i in range(1,len(a_u)-1):
    a_u_h.append("("+str(round(a_u[i],5))+" , "+str(round(a_u[i+1],5))+"]")
a_u_h.append(' ')
for i in range(len(w_u)):
    w_u_h.append(round(w_u[i], 5))
w_u_h.append(sum(w_u))
for u in range(len(P_u)):
    P_u_h.append(round(P_u[i], 5))
P_u_h.append(round(sum(P_u)))
for i in range(len(w_u)):
    abs_u_w_p_h.append(round((abs(w_u[i]-P_u[i])), 5))
abs_u_w_p_h.append(max(abs_u_w_p_h))

x_u_h =[]
n_u_h =[]
for i in range(len(x_u)):
    x_u_h.append(round(x_u[i],5))
x_u_h.append(' ')
for i in range(len(n_u)):
    n_u_h.append(round(n_u[i], 5))
n_u_h.append(sum(n_u))

fig = go.Figure(data=[go.Table(header=dict(values=["Интервалы", 'n_ j' , 'w_ j']),
cells=dict(values=[a_u_h, n_u_h, w_u_h]))
])
fig.show()

fig = go.Figure(data=[go.Table(header=dict(values=["Интервалы", 'w`_ j' , 'p_ j', '|w`_ j - p_ j|']),
cells=dict(values=[a_u_h, w_u_h, P_u_h, abs_u_w_p_h]))
])
fig.show()

fig = go.Figure(data=[go.Table(header=dict(values=["x*_i", 'n`_ j', 'w`_ j']),
cells=dict(values=[x_n_h, n_n_h, w_n_h]))
])
fig.show()

fig = go.Figure(data=[go.Table(header=dict(values=["x*_i", 'n`_ j', 'w`_ j']),
cells=dict(values=[x_i_h, n_i_h, w_i_h]))
])
fig.show()

names = ['Выборочное среднее', 'Выборочная дисперсия', 'Выборочное среднее квадратичное отклонение', 'Выборочная мода', 'Выборочная медиана', 'Выборочный коэффициент асимметрии', 'Выборочный коэффициент эксцесса']

experimental_n = [round(math_exp_n, 5), round(Disp_n, 5), round(stand_dev_n, 5), round(Moda_n, 5), round(Med_n, 5), round(c_as_n, 5), round(c_ex_n, 5)]
theoretical_n = [round(mu, 5), round(Disp_n_t, 5), round(s_d, 5), mu, mu, 0, 0]
absolute_deviation_n = [round(abs(math_exp_n-mu), 5), round(abs(Disp_n - Disp_n_t), 5), round(abs(stand_dev_n-s_d), 5), round(abs(Moda_n-mu), 5), round(abs(Med_n-mu), 5), round(abs(c_as_n-0), 5), round(abs(c_ex_n-0), 5)]
relative_deviation_n = []
for i in range(len(absolute_deviation_n)):
    if(theoretical_n[i] == 0):
        relative_deviation_n.append(' - ')
    else:
        relative_deviation_n.append(round(abs(absolute_deviation_n[i]/theoretical_n[i]) , 5))

fig = go.Figure(data=[go.Table(header=dict(values=['Название показателя', 'Экспериментальное значение' , 'Теоретическое значение', 'Абсолютное отклонение', 'Относительное отклонение']),
cells=dict(values=[names, experimental_n, theoretical_n, absolute_deviation_n, relative_deviation_n]))
])
fig.show()

experimental_i = [round(math_exp_i, 5), round(Disp_i, 5), round(stand_dev_i, 5), round(Moda_i, 5), round(Med_i, 5), round(c_as_i, 5), round(c_ex_i, 5)]
theoretical_i = [round(math_exp_i_t, 5), round(Disp_i_t, 5), round(s_d_i_t, 5), Moda_i_t, round(Med_i_t, 5), c_as_i_t, c_ex_i_t]
absolute_deviation_i = [round(abs(math_exp_i-math_exp_i_t), 5), round(abs(Disp_i - Disp_i_t), 5), round(abs(stand_dev_i-s_d_i_t), 5), round(abs(Moda_i-Moda_i_t), 5), round(abs(Med_i-Med_i_t), 5), round(abs(c_as_i-c_as_i_t), 5), round(abs(c_ex_i-c_ex_i_t), 5)]
relative_deviation_i = []
for i in range(len(absolute_deviation_i)):
    if(theoretical_i[i] == 0):
        relative_deviation_i.append(' - ')
    else:
        relative_deviation_i.append(round(abs(absolute_deviation_i[i]/theoretical_i[i]) , 5))

fig = go.Figure(data=[go.Table(header=dict(values=['Название показателя', 'Экспериментальное значение' , 'Теоретическое значение', 'Абсолютное отклонение', 'Относительное отклонение']),
cells=dict(values=[names, experimental_i, theoretical_i, absolute_deviation_i, relative_deviation_i]))
])
fig.show()

experimental_u = [round(math_exp_u, 5), round(Disp_u, 5), round(stand_dev_u, 5), round(Moda_u, 5), round(Med_u, 5), round(c_as_u, 5), round(c_ex_u, 5)]
theoretical_u = [round(math_exp_u_t, 5), round(Disp_u_t, 5), round(s_d_u_t, 5), round(Moda_u_t, 5), round(Med_u_t, 5), c_as_u_t, c_ex_u_t]
absolute_deviation_u = [round(abs(math_exp_u-math_exp_u_t), 5), round(abs(Disp_u - Disp_u_t), 5), round(abs(stand_dev_u-s_d_u_t), 5), round(abs(Moda_u-Moda_u_t), 5), round(abs(Med_u-Med_u_t), 5), round(abs(c_as_u-c_as_u_t), 5), round(abs(c_ex_u-c_ex_u_t), 5)]
relative_deviation_u = []
for i in range(len(absolute_deviation_u)):
    if(theoretical_u[i] == 0):
        relative_deviation_u.append(' - ')
    else:
        relative_deviation_u.append(round(abs(absolute_deviation_u[i]/theoretical_u[i]) , 5))

fig = go.Figure(data=[go.Table(header=dict(values=['Название показателя', 'Экспериментальное значение' , 'Теоретическое значение', 'Абсолютное отклонение', 'Относительное отклонение']),
cells=dict(values=[names, experimental_u, theoretical_u, absolute_deviation_u, relative_deviation_u]))
])
fig.show()

matrix_v_n_s=[[0] * 20 for i in range(10)]
matrix_v_n_uns=[[0] * 20 for i in range(10)]
matrix_v_i_s=[[0] * 20 for i in range(10)]
matrix_v_i_uns=[[0] * 20 for i in range(10)]
matrix_v_u_s=[[0] * 20 for i in range(10)]
matrix_v_u_uns=[[0] * 20 for i in range(10)]
for i in range(10):
    for j in range(20):
        matrix_v_n_s[i][j]=round(Normal[i+j*10],5)
        matrix_v_n_uns[i][j]=round(Normal_unsort[i+j*10],5)
        matrix_v_i_s[i][j]=round(Indicative[i+j*10],5)
        matrix_v_i_uns[i][j]=round(Indicative_unsort[i+j*10],5)
        matrix_v_u_s[i][j]=round(uniform[i+j*10],5)
        matrix_v_u_uns[i][j]=round(uniform_unsort[i+j*10],5)

workbook = xlsxwriter.Workbook('Normal.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(matrix_v_n_s):
    worksheet.write_column(row, col, data)

workbook.close()

workbook = xlsxwriter.Workbook('Normal_unsort.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(matrix_v_n_uns):
    worksheet.write_column(row, col, data)

workbook.close()

workbook = xlsxwriter.Workbook('Indicative.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(matrix_v_i_s):
    worksheet.write_column(row, col, data)

workbook.close()

workbook = xlsxwriter.Workbook('Indicative_unsort.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(matrix_v_i_uns):
    worksheet.write_column(row, col, data)

workbook.close()

workbook = xlsxwriter.Workbook('uniform.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(matrix_v_u_s):
    worksheet.write_column(row, col, data)

workbook.close()

workbook = xlsxwriter.Workbook('uniform_unsort.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(matrix_v_u_uns):
    worksheet.write_column(row, col, data)

workbook.close()

