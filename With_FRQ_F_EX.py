import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.signal import find_peaks
import scipy.fftpack

plt.style.use('seaborn-poster')



#Function that Calculate Root Mean Square
def rmsValue(arr, First_Index , n ):
    square = 0
    mean = 0.0
    root = 0.0
     
    #Calculate square
    for i in range(First_Index,First_Index + n):
        square += (arr[i]**2)
     
    #Calculate Mean
    mean = (square / (float)(n))
     
    #Calculate Root
    root = math.sqrt(mean)
     
    return root

def crest_factor(x):
    return np.max(np.abs(x))/np.sqrt(np.mean(np.square(x)))

def Average(lst):
    return sum(lst) / len(lst)

def My_FFt(My_Val):    
    yf = scipy.fftpack.fft(My_Val.values)
    return yf


Data_0A_H= pd.read_excel(r'E:\Project\without offset DatA\static\static50%-5A.xlsx')




Va_0_H  = Data_0A_H ['Va']

#Hyper Parameter
SplitFactor=125





Count = int(len(Va_0_H)/SplitFactor)
Va_0_H_S=np.split(Va_0_H, SplitFactor)




RMS_Va_0_H_S = []
kurtosis_Va_0_H_S = []
Crest_factor_Va_0_H_S = []
Skewness_factor_Va_0_H_S = []
Average_Va_0_H_S = []


BeforeFMH_Va_0_H_S = []
AfterFMH_Va_0_H_S = []
BeforeTMH_Va_0_H_S = []
AfterTMH_Va_0_H_S = []
PeakOf_TMF_Va_0_H_S = []
PeakOf_FMF_Va_0_H_S = []

for i in range(SplitFactor):    
    n = len(Va_0_H_S[i])
    print(rmsValue(Va_0_H_S[i], i*Count ,n))
    RMS_Va_0_H_S.append(rmsValue(Va_0_H_S[i], i*Count ,n))

for i in range(SplitFactor):    
    print(kurtosis(Va_0_H_S[i], fisher=False))
    kurtosis_Va_0_H_S.append(kurtosis(Va_0_H_S[i], fisher=False))

for i in range(SplitFactor):    
    print(crest_factor(Va_0_H_S[i]))
    Crest_factor_Va_0_H_S.append(crest_factor(Va_0_H_S[i]))

for i in range(SplitFactor):    
    print(skew(Va_0_H_S[i], axis=0, bias=True))
    Skewness_factor_Va_0_H_S.append(skew(Va_0_H_S[i], axis=0, bias=True))

for i in range(SplitFactor):    
    My_yf= My_FFt(Va_0_H_S[i])
    print(abs((My_yf[11]+My_yf[12])/2))
    BeforeTMH_Va_0_H_S.append(abs((My_yf[11]+My_yf[12])/2))

for i in range(SplitFactor):    
    My_yf= My_FFt(Va_0_H_S[i])
    print(abs((My_yf[12]+My_yf[13])/2))
    AfterTMH_Va_0_H_S.append(abs((My_yf[12]+My_yf[13])/2))


for i in range(SplitFactor):    
    My_yf= My_FFt(Va_0_H_S[i])
    print(abs((My_yf[19]+My_yf[20])/2))
    BeforeFMH_Va_0_H_S.append(abs((My_yf[19]+My_yf[20])/2))


for i in range(SplitFactor):    
    My_yf= My_FFt(Va_0_H_S[i])
    print(abs((My_yf[20]+My_yf[21])/2))
    AfterFMH_Va_0_H_S.append(abs((My_yf[20]+My_yf[21])/2))



for i in range(SplitFactor):    
    My_yf= My_FFt(Va_0_H_S[i])
    print(abs(My_yf[12]))
    PeakOf_TMF_Va_0_H_S.append(abs(My_yf[12]))

for i in range(SplitFactor):    
    My_yf= My_FFt(Va_0_H_S[i])
    print(abs(My_yf[20]))
    PeakOf_FMF_Va_0_H_S.append(abs(My_yf[20]))





# for i in range(SplitFactor):    
#     print(Average(abs(My_FFt(Va_0_H_S[i]))))
#     MeanFrequency_Va_0_H_S.append(Average(abs(My_FFt(Va_0_H_S[i]))))





# creating the DataFrame
Feature_extraction = pd.DataFrame({
                                    
                                    'RMS_Va_0_H_S'            : RMS_Va_0_H_S,
                                    'kurtosis_Va_0_H_S'       : kurtosis_Va_0_H_S,
                                    'Crest_factor_Va_0_H_S'   : Crest_factor_Va_0_H_S,                                    
                                    'Skewness_factor_Va_0_H_S': Skewness_factor_Va_0_H_S,
                                    'BeforeTMH_Va_0_H_S'      : BeforeTMH_Va_0_H_S,  
                                    'AfterTMH_Va_0_H_S'       : AfterTMH_Va_0_H_S,   
                                    'BeforeFMH_Va_0_H_S'      : BeforeFMH_Va_0_H_S,   
                                    'AfterFMH_Va_0_H_S'       : AfterFMH_Va_0_H_S, 
                                    'PeakOf_TMF_Va_0_H_S'     : PeakOf_TMF_Va_0_H_S,
                                    'PeakOf_FMF_Va_0_H_S'     : PeakOf_FMF_Va_0_H_S,
    
                                  })
  
# determining the name of the file
file_name = 'E:\Project\without offset DatA\Feature_ex_static50%-5A.xlsx'
  
# saving the excel
Feature_extraction.to_excel(file_name)
print('DataFrame is written to Excel File successfully.')