import matplotlib.pyplot as plt
import numpy as np
import math
import plotly.graph_objects as go
import copy
import xlsxwriter
import pandas as pd
import scipy
from scipy.stats import norm

a=0.33
sigma=0.866
sigma_2=sigma2

df=pd.read_excel(selection_1.xlsx,Лист1)
Selection_1_unsort = []
for i in range(len(df.columns))
    Selection_1_unsort.extend(df[df.columns[i]].tolist())
    
df_2=pd.read_excel(selection_3.xlsx,Лист1)
Selection_3_1_unsort = []
Selection_3_2_unsort = []
Selection_3_3_unsort = []
Selection_3_1_unsort.extend(df_2[df_2.columns[0]].tolist())
Selection_3_2_unsort.extend(df_2[df_2.columns[1]].tolist())
Selection_3_3_unsort.extend(df_2[df_2.columns[2]].tolist())

Selection_3_unsort = [df_2[df_2.columns[0]].tolist(),df_2[df_2.columns[1]].tolist(), df_2[df_2.columns[2]].tolist()]

Selection_1 = copy.deepcopy(Selection_1_unsort)
Selection_1.sort()
Selection_3_1 = copy.deepcopy(Selection_3_1_unsort)
Selection_3_1.sort()
Selection_3_2 = copy.deepcopy(Selection_3_2_unsort)
Selection_3_2.sort()
Selection_3_3 = copy.deepcopy(Selection_3_3_unsort)
Selection_3_3.sort()

N=len(Selection_1)
alpha=0.05
mu_1=sum(Selection_1)N
C=a+sigmanorm.ppf(1-alpha)math.sqrt(N)

print(mu_1)
print(C)

N_3_1=len(Selection_3_1)
N_3_2=len(Selection_3_2)
N_3_3=len(Selection_3_3)

x_2_3_1=sum(map(lambda xxx,Selection_3_1))N_3_1
x_2_3_2=sum(map(lambda xxx,Selection_3_2))N_3_2
x_2_3_3=sum(map(lambda xxx,Selection_3_3))N_3_3

x_3_1=sum(Selection_3_1)N_3_1
x_3_2=sum(Selection_3_2)N_3_2
x_3_3=sum(Selection_3_3)N_3_3

S_2_3_1=N_3_1(x_2_3_1-x_3_12)(N_3_1-1)
S_2_3_2=N_3_2(x_2_3_2-x_3_22)(N_3_2-1)
S_2_3_3=N_3_3(x_2_3_3-x_3_32)(N_3_3-1)-1)

T_1_2=(x_3_1-x_3_2)math.sqrt(N_3_1N_3_2(N_3_1+N_3_2-2)(N_3_2+N_3_1))math.sqrt(S_2_3_1(N_3_1-1)+S_2_3_2(N_3_2-1))
T_1_3=(x_3_1-x_3_3)math.sqrt(N_3_1N_3_3(N_3_1+N_3_3-2)(N_3_1+N_3_3))math.sqrt(S_2_3_1(N_3_1-1)+S_2_3_3(N_3_3-1))
T_2_3=(x_3_2-x_3_3)math.sqrt(N_3_2N_3_3(N_3_2+N_3_3-2)(N_3_2+N_3_3))math.sqrt(S_2_3_2(N_3_2-1)+S_2_3_3(N_3_3-1))

x=1-alpha2
n=N_3_1+N_3_2-2
t_kr=scipy.stats.t.ppf(x,n)

print(abs(T_1_2)=t_kr)
print(abs(T_1_3)=t_kr)
print(abs(T_2_3)=t_kr)

u_mean=(sum(Selection_3_1)+sum(Selection_3_2)+sum(Selection_3_3))(N_3_13)

S_common=sum(map(lambda x(x-u_mean)2,Selection_3_1))
S_common+=sum(map(lambda x(x-u_mean)2,Selection_3_2))
S_common+=sum(map(lambda x(x-u_mean)2,Selection_3_3))

S_fact=N_3_1((x_3_1-u_mean)2)((x_3_2-u_mean)2)((x_3_3-u_mean)2)

S_residual=S_common-S_fact

m=3
k_1=m-1
k_2=m(N_3_1-1)
S_2_fact=S_factk_1
S_2_residual=S_residualk_2
F_N_M=S_2_factS_2_residual

z_a=scipy.stats.f.isf(alpha,k_1,k_2)

print(F_N_M=z_a)

pval=scipy.stats.f_oneway(Selection_3_unsort[0],Selection_3_unsort[1],Selection_3_unsort[2])

pval_1_2=scipy.stats.ttest_ind(Selection_3_unsort[0],Selection_3_unsort[1])
pval_1_3=scipy.stats.ttest_ind(Selection_3_unsort[0],Selection_3_unsort[2])
pval_2_3=scipy.stats.ttest_ind(Selection_3_unsort[1],Selection_3_unsort[2])

pval_w_1_2=scipy.stats.ttest_ind(Selection_3_unsort[0],Selection_3_unsort[1], equal_var=False)
pval_w_1_3=scipy.stats.ttest_ind(Selection_3_unsort[0],Selection_3_unsort[2], equal_var=False)
pval_w_2_3=scipy.stats.ttest_ind(Selection_3_unsort[1],Selection_3_unsort[2], equal_var=False)

S_2_1_1_2=max(S_2_3_1,S_2_3_2)
S_2_1_1_3=max(S_2_3_1,S_2_3_3)
S_2_1_2_3=max(S_2_3_2,S_2_3_3)
S_2_2_1_2=min(S_2_3_1,S_2_3_2)
S_2_2_1_3=min(S_2_3_1,S_2_3_3)
S_2_2_2_3=min(S_2_3_2,S_2_3_3)

F_1_2=S_2_1_1_2S_2_2_1_2
F_1_3=S_2_1_1_3S_2_2_1_3
F_2_3=S_2_1_2_3S_2_2_2_3

k_4_5=N_3_1-1
z_a_4_5=scipy.stats.f.isf(alpha2, k_4_5, k_4_5)

pval_bar=scipy.stats.bartlett(Selection_3_unsort[0],Selection_3_unsort[1],Selection_3_unsort[2])

pval_lev_1_2=scipy.stats.levene(Selection_3_unsort[0],Selection_3_unsort[1])
pval_lev_1_3=scipy.stats.levene(Selection_3_unsort[0],Selection_3_unsort[2])
pval_lev_2_3=scipy.stats.levene(Selection_3_unsort[1],Selection_3_unsort[2])

