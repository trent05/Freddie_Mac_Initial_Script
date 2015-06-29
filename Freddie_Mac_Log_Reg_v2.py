import numpy as np
import pandas as pd
import statsmodels.api as sm
import pdb
import math
from sklearn import metrics

df = pd.read_csv('historical_data1_Q32008_v4.csv')

def one(n):
 return 1
df['Constant'] = df['First_Pay_Date'].apply(one)

dfLP = pd.get_dummies(df['Loan_Purpose'], prefix='Loan_Purp')
dfFTH = pd.get_dummies(df['First_Time_Homebuyer'], prefix='FTH')
dfNoU = pd.get_dummies(df['Number_of_Units'], prefix='NoU')
dfOS = pd.get_dummies(df['Occupancy_Status'], prefix='OS')
dfCH = pd.get_dummies(df['Channel'], prefix='CH')
dfNoB = pd.get_dummies(df['Number_of_Borrowers'], prefix='NoB')


df = pd.merge(df, dfLP, left_index=True, right_index=True)
df = pd.merge(df, dfFTH, left_index=True, right_index=True)
df = pd.merge(df, dfNoU, left_index=True, right_index=True)
df = pd.merge(df, dfOS, left_index=True, right_index=True)
df = pd.merge(df, dfCH, left_index=True, right_index=True)
df = pd.merge(df, dfNoB, left_index=True, right_index=True)


a ='Loan_Purp_C', 'Loan_Purp_N', 'Loan_Purp_P'
b = 'FTH_N', 'FTH_Y', 'FTH_NA'
c = 'NoU_1', 'NoU_2', 'NoU_3', 'NoU_4'
d = 'OS_I', 'OS_O', 'OS_S'
e = 'CH_B', 'CH_C', 'CH_R', 'CH_T'
f = 'NoB_1', 'NoB_2', 'NoB_NA'


Ind_Vars = ['Credit_Score', 'Orig_Combined_LTV', 'Orig_Debt_to_Income', 'Orig_Int_Rate', 'Loan_Purp_C', 'Loan_Purp_N', 'Loan_Purp_P', 'FTH_N', 'FTH_Y', 'NoU_1', 'NoU_2', 'NoU_3', 'NoU_4', 'OS_I', 'OS_O', 'OS_S', 'CH_B', 'CH_C', 'CH_R', 'CH_T', 'NoB_1.0', 'NoB_2.0']
#print df.head()


logit = sm.Logit(df['Deliquency'], df[Ind_Vars])
result = logit.fit()
coeff = result.params
#print result.summary()

y = df['Deliquency']
predicted = result.predict(df['Deliquency'])
expected = y
#print result
#print predicted
#print expected
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
