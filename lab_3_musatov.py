import matplotlib.pyplot as plt
import numpy as np
import math
import plotly.graph_objects as go
import copy
import xlsxwriter
import pandas as pd

df=pd.read_excel("first_selection.xlsx","Лист1")
df_2=pd.read_excel("second_selection.xlsx","Лист1")
Selection_1_unsort = []
Selection_2_unsort = []
for i in range(len(df.columns)):
    Selection_1_unsort.extend(df[df.columns[i]].tolist())
    Selection_2_unsort.extend(df_2[df_2.columns[i]].tolist())

print(df.columns[0])
print(len(Selection_2_unsort))

Selection_1 = copy.deepcopy(Selection_1_unsort)
Selection_1.sort()
Selection_2 = copy.deepcopy(Selection_2_unsort)
Selection_2.sort()

N=len(Selection_1_unsort)
m=1+int(math.log(N,2))
n_0=min(Selection_1_unsort)
n_m=max(Selection_1_unsort)
d_n=n_m-n_0
a_n=[n_0]
for i in range(m-1):
    a_n.append(a_n[i]+d_n/m)
a_n.append(n_m)

print(len(a_n))
for i in range(len(a_n)):
    print(a_n[i])

n_n=[]
counter = 0
k=1
helper = a_n[k]
for i in range(len(Selection_1)):
    if Selection_1[i]<=helper:
        counter += 1 
    else:
        n_n.append(counter)
        k += 1
        helper = a_n[k]
        if (Selection_1[i]<=helper):
            counter=1
        else: counter=0
n_n.append(counter)

w_n=[]
for i in range(len(n_n)):
    w_n.append(n_n[i]/N)
    
x_n = []
for i in range(len(a_n)-1):
    x_n.append((a_n[i]+a_n[i+1])/2)
    
math_exp_n=0
for i in range(len(x_n)):
    math_exp_n+=(x_n[i]*w_n[i])
print(math_exp_n)

h_n=(n_m-n_0)/m

Disp_n=0
for i in range(len(x_n)):
    Disp_n+=w_n[i]*((x_n[i])**2)
Disp_n-=((h_n**2)/12+math_exp_n**2)
print(Disp_n)

stand_dev_n=math.sqrt(Disp_n)

print(stand_dev_n)

#функция распределения
def normal_prob(x, miu, stdev):
    return 0.5 * (1 + math.erf((x-miu)/(stdev * 2**0.5)))

def c_norm_prob(x, miu, stdev):
    return (x-miu)/(stdev)

def density_normal_prob_t(t):
    return math.exp(-(t**2)/2)/math.sqrt(2*math.pi)

P_n = []
C_n_prob = []
n_prob = []
F_n_prob = []
P_n.append(" - ")
for i in range (2):
    C_n_prob.append(c_norm_prob(a_n[i], math_exp_n,stand_dev_n))
    n_prob.append(density_normal_prob_t(C_n_prob[i])/stand_dev_n)
    F_n_prob.append(normal_prob(a_n[i], math_exp_n,stand_dev_n))

P_n.append(F_n_prob[1])
for i in range(2,len(a_n)-1):
    C_n_prob.append(c_norm_prob(a_n[i], math_exp_n,stand_dev_n))
    n_prob.append(density_normal_prob_t(C_n_prob[i])/stand_dev_n)
    F_n_prob.append(normal_prob(a_n[i], math_exp_n,stand_dev_n))
    P_n.append(F_n_prob[i]-F_n_prob[i-1])
    
C_n_prob.append(c_norm_prob(a_n[len(a_n)-1], math_exp_n,stand_dev_n))
n_prob.append(density_normal_prob_t(C_n_prob[len(a_n)-1])/stand_dev_n)
F_n_prob.append(normal_prob(a_n[len(a_n)-1], math_exp_n,stand_dev_n))
P_n.append(1-F_n_prob[len(a_n)-2])

k=list(range(0, m+1))

P_n_h = []
C_n_prob_h = []
n_prob_h = []
F_n_prob_h = []
a_n_h=[]

P_n_h.append(P_n[0])
C_n_prob_h.append(round(C_n_prob[0],5))
n_prob_h.append(round(n_prob[0],5))
F_n_prob_h.append(round(F_n_prob[0],5))
a_n_h.append(round(a_n[0],5))

for i in range(1,len(P_n)):
    P_n_h.append(round(P_n[i],5))
    C_n_prob_h.append(round(C_n_prob[i],5))
    n_prob_h.append(round(n_prob[i],5))
    F_n_prob_h.append(round(F_n_prob[i],5))
    a_n_h.append(round(a_n[i],5))
    
a_n_h.append(' ')
C_n_prob_h.append(' ')
n_prob_h.append(' ')
F_n_prob_h.append(' ')
P_n_h.append(round(sum(P_n[1:len(P_n)]),5))

fig = go.Figure(data=[go.Table(header=dict(values=["k", "a_k" , "(a_k-a)/sig", "fi((a_k-a)/sig)/sig", "F((a_k-a)/sig)", "p_k" ]),
cells=dict(values=[k, a_n_h, C_n_prob_h, n_prob_h, F_n_prob_h, P_n_h ]))
])
fig.show()

i_n=[]
w_n_h=[]
abs_n_w_p_h = []
P_n_h_1=P_n_h[1:len(P_n_h)]
n_w_p=[]
k_h=[]

h="["+str(round(a_n[0],5))+" , "+str(round(a_n[1],5))+"]"
i_n.append(h)
for i in range(1,len(a_n)-1):
    i_n.append("("+str(round(a_n[i],5))+" , "+str(round(a_n[i+1],5))+"]")
i_n.append(' ')


for i in range(len(w_n)):
    w_n_h.append(round(w_n[i], 5))
w_n_h.append(sum(w_n))

for i in range(len(w_n)):
    abs_n_w_p_h.append(round((abs(w_n[i]-P_n_h_1[i])), 5))
abs_n_w_p_h.append(max(abs_n_w_p_h))

for i in range(len(w_n)):
    n_w_p.append(round(N*((w_n[i]-P_n_h_1[i])**2)/P_n_h_1[i],5))
n_w_p.append(sum(n_w_p))

k_h=k[1:len(k)]
fig = go.Figure(data=[go.Table(header=dict(values=["k", "Интервалы", "w_k" , "p_k", "|w_k - p_k|" , "N(w_k-p_k)^2/p_k"]),
cells=dict(values=[k_h, i_n, w_n_h, P_n_h_1, abs_n_w_p_h, n_w_p]))
])
fig.show()

density_x_n = np.linspace(n_0, n_m, 200)
density_y_n = []
for i in range(len(density_x_n)):
    density_y_n.append(density_normal_prob_t(c_norm_prob(density_x_n[i],math_exp_n,stand_dev_n)))

fig = plt.figure(figsize=(20, 10))
plt.hist(Selection_1,bins=m, density=True)
plt.plot( a_n, n_prob,'r', linewidth=1)
plt.title("Гистограмма распределения первой выборки и график плотности нормального распределения",fontsize =15)
plt.yticks(np.arange(0.0, 0.5, step=0.025), fontsize =12)
plt.xticks(np.arange(round(Selection_1[0],1)-0.1, round(Selection_1[len(Selection_1)-1]+0.2,1), step=0.2), fontsize =12)
plt.savefig('selection_1_hist_t.jpg')

fig = plt.figure(figsize=(20, 10))
plt.hist(Selection_1,bins=m, density=True)
plt.plot( density_x_n, density_y_n,'r', linewidth=1)
plt.title("Гистограмма распределения первой выборки и график плотности нормального распределения",fontsize =15)
plt.yticks(np.arange(0.0, 0.5, step=0.025), fontsize =12)
plt.xticks(np.arange(round(Selection_1[0],1)-0.1, round(Selection_1[len(Selection_1)-1]+0.2,1), step=0.2), fontsize =12)
plt.savefig('selection_1_hist.jpg')

x_n_h =[]
n_n_h =[]

for i in range(len(x_n)):
    x_n_h.append(round(x_n[i],5))
x_n_h.append(' ')

for i in range(len(n_n)):
    n_n_h.append(round(n_n[i], 5))
n_n_h.append(sum(n_n))

fig = go.Figure(data=[go.Table(header=dict(values=["Интервалы", "x*_k", "n_k" , "w_k"]),
cells=dict(values=[i_n, x_n, n_n_h, w_n_h]))
])
fig.show()

N_2=len(Selection_2)
m_2=1+int(math.log(N,2))
u_0=min(Selection_2)
u_m=max(Selection_2)
d_u=u_m-u_0
a_u=[u_0]
for i in range(m-1):
    a_u.append(a_u[i]+d_u/m_2)
a_u.append(u_m)
n_u=[]
counter = 0
k=1
helper = a_u[k]
for i in range(len(Selection_2)):
    if Selection_2[i]<=helper:
        counter += 1 
    else:
        n_u.append(counter)
        k += 1
        helper = a_u[k]
        if (Selection_2[i]<=helper):
            counter=1
        else: counter=0
n_u.append(counter)

w_u=[]
for i in range(len(n_u)):
    w_u.append(n_u[i]/N_2)
    
x_u = []
for i in range(len(a_u)-1):
    x_u.append((a_u[i]+a_u[i+1])/2)
    
math_exp_u=0
for i in range(len(x_u)):
    math_exp_u+=(x_u[i]*w_u[i])
print(math_exp_u)

h_u=(u_m-u_0)/m_2

Disp_u=0
for i in range(len(x_u)):
    Disp_u+=w_u[i]*((x_u[i])**2)
Disp_u-=((h_u**2)/12+math_exp_u**2)
print(Disp_u)

stand_dev_u=math.sqrt(Disp_u)

n_u_h =[]

for i in range(len(n_u)):
    n_u_h.append(round(n_u[i], 5))
n_u_h.append(sum(n_u))

u_w_p=[]
p_k_u=1/m_2

for i in range(len(w_u)):
    u_w_p.append(round(N*((w_u[i]-p_k_u)**2)/p_k_u,5))
u_w_p.append(sum(u_w_p))

k_u=list(range(1, m+1))

i_u = []
w_u_h = []
abs_u_w_p_h = []

h="["+str(round(a_u[0],5))+" , "+str(round(a_u[1],5))+"]"
i_u.append(h)
for i in range(1,len(a_u)-1):
    i_u.append("("+str(round(a_u[i],5))+" , "+str(round(a_u[i+1],5))+"]")
i_u.append(' ')

for i in range(len(w_u)):
    w_u_h.append(round(w_u[i], 5))
w_u_h.append(sum(w_u))

for i in range(len(w_u)):
    abs_u_w_p_h.append(round((abs(w_n[i]-p_k_u)), 5))
abs_u_w_p_h.append(max(abs_u_w_p_h))

P_u_h=[p_k_u for i in range(m_2)]
P_u_h.append(sum(P_u_h))

fig = go.Figure(data=[go.Table(header=dict(values=["Интервалы", "x*_k", "n_k" , "w_k"]),
cells=dict(values=[i_u, x_u, n_u_h, w_u_h]))
])
fig.show()

fig = go.Figure(data=[go.Table(header=dict(values=["k", "Интервалы", "w_k" , "p_k", "|w_k - p_k|" , "N(w_k-p_k)^2/p_k"]),
cells=dict(values=[k_u, i_u, w_u_h, P_u_h, abs_u_w_p_h, u_w_p]))
])
fig.show()

h_x=[u_0, u_m]
h_y=[1/(u_m-u_0), 1/(u_m-u_0)]

fig = plt.figure(figsize=(20, 10))
plt.hist(Selection_2,bins=m, density=True)
plt.plot( h_x, h_y,'r', linewidth=1)
plt.title("Гистограмма распределения первой выборки и график плотности нормального распределения",fontsize =15)
plt.yticks(np.arange(0.0, 0.25, step=0.025), fontsize =12)
plt.xticks(np.arange(round(Selection_2[0],1)-0.1, round(Selection_2[len(Selection_2)-1]+0.2,1), step=0.2), fontsize =12)
plt.savefig('selection_2_hist.jpg')

S=[1/N_2]
now_h= 1/(u_m-u_0)
for i in range(1,N_2):
    S.append(S[i-1]+(1/N_2))

x_2=np.linspace(u_0, u_m,200)   
y_2 =[]
for i in range(len(x_2)):
    y_2.append((x_2[i]-u_0)*now_h)

fig = plt.figure(figsize=(20, 10))
plt.hlines(0, Selection_2[0]-0.05, Selection_2[0])
for i in range(len(S)-1):
    plt.hlines(S[i], Selection_2[i], Selection_2[i+1])
plt.hlines(S[i+1], Selection_2[i+1], Selection_2[i+1]+0.05)    
plt.plot(x_2, y_2, color = 'r')
plt.title('Эмпирическая функция выборочного распределения',fontsize =15)
plt.ylabel('s - sum(w(x)) ', fontsize =15)
plt.yticks(np.arange(0.0, 1.05, step=0.05), fontsize =12)
plt.xticks(np.arange(round(Selection_2[0],1), round(Selection_2[len(Selection_2)-1]+0.2,1), step=0.2), fontsize =12)
plt.xlabel('x - value', fontsize =15)
plt.savefig('uniform_emp.jpg')

matrix_1_s=[[0] * 20 for i in range(10)]
matrix_2_s=[[0] * 20 for i in range(10)]
for i in range(10):
    for j in range(20):
        matrix_1_s[i][j]=round(Selection_1[i+j*10],5)
        matrix_2_s[i][j]=round(Selection_2[i+j*10],5)

workbook = xlsxwriter.Workbook('Selection_1_sort.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(matrix_1_s):
    worksheet.write_column(row, col, data)

workbook.close()

workbook = xlsxwriter.Workbook('Selection_2_sort.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(matrix_2_s):
    worksheet.write_column(row, col, data)

workbook.close()

k_h_h=copy.deepcopy(k_h)
k_h_h.append(" ")
k_u_h=copy.deepcopy(k_u)
k_u_h.append(" ")

matrix_1_t=[[0] * 9 for i in range(6)]
matrix_2_t=[[0] * 9 for i in range(6)]
for i in range(9):
        matrix_1_t[0][i] = k_h_h[i]
        matrix_1_t[1][i] = i_n[i]
        matrix_1_t[2][i] = w_n_h[i]
        matrix_1_t[3][i] = P_n_h_1[i]
        matrix_1_t[4][i] = abs_n_w_p_h[i]
        matrix_1_t[5][i] = n_w_p[i]
        matrix_2_t[0][i] = k_u_h[i]
        matrix_2_t[1][i] = i_u[i]
        matrix_2_t[2][i] = w_u_h[i]
        matrix_2_t[3][i] = P_u_h[i]
        matrix_2_t[4][i] = abs_u_w_p_h[i]
        matrix_2_t[5][i] = u_w_p[i]

workbook = xlsxwriter.Workbook('1_table_1_2.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(matrix_1_t):
    worksheet.write_column(row, col, data)

workbook.close()

k=list(range(0, m+1))
k.append(" ")

matrix_1_t_1=[[0] * 10 for i in range(6)]
for i in range(10):
        matrix_1_t_1[0][i] = k[i]
        matrix_1_t_1[1][i] = a_n_h[i]
        matrix_1_t_1[2][i] = C_n_prob_h[i]
        matrix_1_t_1[3][i] = n_prob_h[i]
        matrix_1_t_1[4][i] = F_n_prob_h[i]
        matrix_1_t_1[5][i] = P_n_h[i]

workbook = xlsxwriter.Workbook('1_table_1_1.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(matrix_1_t_1):
    worksheet.write_column(row, col, data)

workbook.close()

workbook = xlsxwriter.Workbook('2_table_2_1.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(matrix_2_t):
    worksheet.write_column(row, col, data)

workbook.close()

def helper_func( x, u_0, u_m):
    return ((x-u_0)/(u_m-u_0))

x_j = abs(helper_func(Selection_2[0],u_0, u_m)-S[0])
index_j = 0
for i in range(1,len(S)):
    max_now_f_0=max(abs(helper_func(Selection_2[i],u_0, u_m)-S[i-1]), abs(helper_func(Selection_2[i],u_0, u_m)-S[i]))
    if(x_j<max_now_f_0):
        x_j = max_now_f_0
        index_j = i

index_j

Selection_2[index_j]
S[index_j]

helper_func(Selection_2[index_j],u_0, u_m)

helper_func(Selection_2[index_j-1],u_0, u_m)

x_j

x_j*math.sqrt(200)

