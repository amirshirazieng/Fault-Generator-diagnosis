import tensorflow as tf
from keras import layers, optimizers, losses
from keras.layers import Dense, Dropout
import keras
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import re
import scipy.io
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.layers.convolutional import MaxPooling2D,Conv1D,MaxPooling1D
from keras.layers.core import Flatten
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Activation
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import keras

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-poster')

print(tf.__version__)
print(keras.__version__)



files_name = {}
 
def Gen_xkhz():
  # Healthy
  files_name["He_Ia0A"]       = "He_Ia0A"
  files_name["He_Ib0A"]       = "He_Ib0A"
  files_name["He_If0A"]       = "He_If0A"
  files_name["He_Va0A"]       = "He_Va0A"
  files_name["He_Vb0A"]       = "He_Vb0A"

  files_name["He_Ia1A"]       = "He_Ia1A"
  files_name["He_Ib1A"]       = "He_Ib1A"
  files_name["He_If1A"]       = "He_If1A"
  files_name["He_Va1A"]       = "He_Va1A"
  files_name["He_Vb1A"]       = "He_Vb1A"

  files_name["He_Ia2A"]       = "He_Ia2A"
  files_name["He_Ib2A"]       = "He_Ib2A"
  files_name["He_If2A"]       = "He_If2A"
  files_name["He_Va2A"]       = "He_Va2A"
  files_name["He_Vb2A"]       = "He_Vb2A"
    
  files_name["He_Ia3A"]       = "He_Ia3A"  
  files_name["He_Ib3A"]       = "He_Ib3A"  
  files_name["He_If3A"]       = "He_If3A"  
  files_name["He_Va3A"]       = "He_Va3A"  
  files_name["He_Vb3A"]       = "He_Vb3A"  

  files_name["He_Ia4A"]       = "He_Ia4A" 
  files_name["He_Ib4A"]       = "He_Ib4A" 
  files_name["He_If4A"]       = "He_If4A" 
  files_name["He_Va4A"]       = "He_Va4A" 
  files_name["He_Vb4A"]       = "He_Vb4A" 
  
  files_name["He_Ia5A"]       = "He_Ia5A"
  files_name["He_Ib5A"]       = "He_Ib5A"  
  files_name["He_If5A"]       = "He_If5A"  
  files_name["He_Va5A"]       = "He_Va5A"  
  files_name["He_Vb5A"]       = "He_Vb5A"  

  # morakab  
  files_name["Fa_M_Ia0A"]       = "Fa_M_Ia0A"
  files_name["Fa_M_Ib0A"]       = "Fa_M_Ib0A"
  files_name["Fa_M_If0A"]       = "Fa_M_If0A"
  files_name["Fa_M_Va0A"]       = "Fa_M_Va0A"
  files_name["Fa_M_Vb0A"]       = "Fa_M_Vb0A"
  
  files_name["Fa_M_Ia1A"]       = "Fa_M_Ia1A"
  files_name["Fa_M_Ib1A"]       = "Fa_M_Ib1A"
  files_name["Fa_M_If1A"]       = "Fa_M_If1A"
  files_name["Fa_M_Va1A"]       = "Fa_M_Va1A"
  files_name["Fa_M_Vb1A"]       = "Fa_M_Vb1A"

  files_name["Fa_M_Ia2A"]       = "Fa_M_Ia2A"
  files_name["Fa_M_Ib2A"]       = "Fa_M_Ib2A"
  files_name["Fa_M_If2A"]       = "Fa_M_If2A"
  files_name["Fa_M_Va2A"]       = "Fa_M_Va2A"
  files_name["Fa_M_Vb2A"]       = "Fa_M_Vb2A"

  files_name["Fa_M_Ia3A"]       = "Fa_M_Ia3A" 
  files_name["Fa_M_Ib3A"]       = "Fa_M_Ib3A" 
  files_name["Fa_M_If3A"]       = "Fa_M_If3A" 
  files_name["Fa_M_Va3A"]       = "Fa_M_Va3A" 
  files_name["Fa_M_Vb3A"]       = "Fa_M_Vb3A" 
    
  files_name["Fa_M_Ia4A"]       = "Fa_M_Ia4A"
  files_name["Fa_M_Ib4A"]       = "Fa_M_Ib4A"
  files_name["Fa_M_If4A"]       = "Fa_M_If4A"
  files_name["Fa_M_Va4A"]       = "Fa_M_Va4A"
  files_name["Fa_M_Vb4A"]       = "Fa_M_Vb4A"

  files_name["Fa_M_Ia5A"]       = "Fa_M_Ia5A"
  files_name["Fa_M_Ib5A"]       = "Fa_M_Ib5A"
  files_name["Fa_M_If5A"]       = "Fa_M_If5A"
  files_name["Fa_M_Va5A"]       = "Fa_M_Va5A"
  files_name["Fa_M_Vb5A"]       = "Fa_M_Vb5A"

  # Dynamic 
  files_name["Fa_D_Ia0A"]       = "Fa_D_Ia0A"
  files_name["Fa_D_Ib0A"]       = "Fa_D_Ib0A"
  files_name["Fa_D_If0A"]       = "Fa_D_If0A"
  files_name["Fa_D_Va0A"]       = "Fa_D_Va0A"
  files_name["Fa_D_Vb0A"]       = "Fa_D_Vb0A"

  files_name["Fa_D_Ia1A"]       = "Fa_D_Ia1A"
  files_name["Fa_D_Ib1A"]       = "Fa_D_Ib1A"
  files_name["Fa_D_If1A"]       = "Fa_D_If1A"
  files_name["Fa_D_Va1A"]       = "Fa_D_Va1A"
  files_name["Fa_D_Vb1A"]       = "Fa_D_Vb1A"

  files_name["Fa_D_Ia2A"]       = "Fa_D_Ia2A"
  files_name["Fa_D_Ib2A"]       = "Fa_D_Ib2A"
  files_name["Fa_D_If2A"]       = "Fa_D_If2A"
  files_name["Fa_D_Va2A"]       = "Fa_D_Va2A"
  files_name["Fa_D_Vb2A"]       = "Fa_D_Vb2A"
    
  files_name["Fa_D_Ia3A"]       = "Fa_D_Ia3A"
  files_name["Fa_D_Ib3A"]       = "Fa_D_Ib3A"
  files_name["Fa_D_If3A"]       = "Fa_D_If3A"
  files_name["Fa_D_Va3A"]       = "Fa_D_Va3A"
  files_name["Fa_D_Vb3A"]       = "Fa_D_Vb3A"


  files_name["Fa_D_Ia4A"]       = "Fa_D_Ia4A"
  files_name["Fa_D_Ib4A"]       = "Fa_D_Ib4A"
  files_name["Fa_D_If4A"]       = "Fa_D_If4A"
  files_name["Fa_D_Va4A"]       = "Fa_D_Va4A"
  files_name["Fa_D_Vb4A"]       = "Fa_D_Vb4A"

    
  files_name["Fa_D_Ia5A"]       = "Fa_D_Ia5A"
  files_name["Fa_D_Ib5A"]       = "Fa_D_Ib5A"
  files_name["Fa_D_If5A"]       = "Fa_D_If5A"
  files_name["Fa_D_Va5A"]       = "Fa_D_Va5A"
  files_name["Fa_D_Vb5A"]       = "Fa_D_Vb5A"

   
  return files_name

acquisitions = {}





Data_0A_H= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\Healthy-0A.xlsx')
print(Data_0A_H.head())

Time = Data_0A_H ['Time']

Ia_0_H = Data_0A_H ['Ia']
Ib_0_H = Data_0A_H ['Ib']
If_0_H = Data_0A_H ['If']
Va_0_H = Data_0A_H ['Va']
Vb_0_H = Data_0A_H ['Vb']

################################# Healthy
Data_1A_H= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\Healthy-1A.xlsx')
print(Data_1A_H.head())

Time = Data_1A_H ['Time']

Ia_1_H = Data_1A_H ['Ia']
Ib_1_H = Data_1A_H ['Ib']
If_1_H = Data_1A_H ['If']
Va_1_H = Data_1A_H ['Va']
Vb_1_H = Data_1A_H ['Vb']
################################# Healthy
Data_2A_H= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\Healthy-2A.xlsx')
print(Data_2A_H.head())

Time = Data_2A_H ['Time']

Ia_2_H = Data_2A_H ['Ia']
Ib_2_H = Data_2A_H ['Ib']
If_2_H = Data_2A_H ['If']
Va_2_H = Data_2A_H ['Va']
Vb_2_H = Data_2A_H ['Vb']
################################# Healthy
Data_3A_H= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\Healthy-3A.xlsx')
print(Data_3A_H.head())

Time = Data_3A_H ['Time']

Ia_3_H = Data_3A_H ['Ia']
Ib_3_H = Data_3A_H ['Ib']
If_3_H = Data_3A_H ['If']
Va_3_H = Data_3A_H ['Va']
Vb_3_H = Data_3A_H ['Vb']
################################# Healthy
Data_4A_H= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\Healthy-4A.xlsx')
print(Data_4A_H.head())

Time = Data_4A_H ['Time']

Ia_4_H = Data_4A_H ['Ia']
Ib_4_H = Data_4A_H ['Ib']
If_4_H = Data_4A_H ['If']
Va_4_H = Data_4A_H ['Va']
Vb_4_H = Data_4A_H ['Vb']
################################# Healthy
Data_5A_H= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\Healthy-5A.xlsx')
print(Data_5A_H.head())

Time = Data_5A_H ['Time']

Ia_5_H = Data_5A_H ['Ia']
Ib_5_H = Data_5A_H ['Ib']
If_5_H = Data_5A_H ['If']
Va_5_H = Data_5A_H ['Va']
Vb_5_H = Data_5A_H ['Vb']





################################# morakab
Data_0A_M= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\morakab50%-0A.xlsx')
print(Data_0A_M.head())

Time = Data_0A_M ['Time']

Ia_0_M = Data_0A_M ['Ia']
Ib_0_M = Data_0A_M ['Ib']
If_0_M = Data_0A_M ['If']
Va_0_M = Data_0A_M ['Va']
Vb_0_M = Data_0A_M ['Vb']

################################# morakab
Data_1A_M= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\morakab50%-1A.xlsx')
print(Data_1A_M.head())

Time = Data_1A_M ['Time']

Ia_1_M = Data_1A_M ['Ia']
Ib_1_M = Data_1A_M ['Ib']
If_1_M = Data_1A_M ['If']
Va_1_M = Data_1A_M ['Va']
Vb_1_M = Data_1A_M ['Vb']
################################# morakab
Data_2A_M= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\morakab50%-2A.xlsx')
print(Data_2A_M.head())

Time = Data_2A_M ['Time']

Ia_2_M = Data_2A_M ['Ia']
Ib_2_M = Data_2A_M ['Ib']
If_2_M = Data_2A_M ['If']
Va_2_M = Data_2A_M ['Va']
Vb_2_M = Data_2A_M ['Vb']

################################# morakab
Data_3A_M= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\morakab50%-3A.xlsx')
print(Data_3A_M.head())

Time = Data_3A_M ['Time']

Ia_3_M = Data_3A_M ['Ia']
Ib_3_M = Data_3A_M ['Ib']
If_3_M = Data_3A_M ['If']
Va_3_M = Data_3A_M ['Va']
Vb_3_M = Data_3A_M ['Vb']

################################# morakab
Data_4A_M= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\morakab50%-4A.xlsx')
print(Data_4A_M.head())

Time = Data_4A_M ['Time']

Ia_4_M = Data_4A_M ['Ia']
Ib_4_M = Data_4A_M ['Ib']
If_4_M = Data_4A_M ['If']
Va_4_M = Data_4A_M ['Va']
Vb_4_M = Data_4A_M ['Vb']

################################# morakab
Data_5A_M= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\morakab50%-5A.xlsx')
print(Data_5A_M.head())

Time = Data_5A_M ['Time']

Ia_5_M = Data_5A_M ['Ia']
Ib_5_M = Data_5A_M ['Ib']
If_5_M = Data_5A_M ['If']
Va_5_M = Data_5A_M ['Va']
Vb_5_M = Data_5A_M ['Vb']





################################# Dynamic
Data_0A_D= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\50%Dynamic-0A.xlsx')
print(Data_0A_D.head())

Time = Data_0A_D ['Time']

Ia_0_D = Data_0A_D ['Ia']
Ib_0_D = Data_0A_D ['Ib']
If_0_D = Data_0A_D ['If']
Va_0_D = Data_0A_D ['Va']
Vb_0_D = Data_0A_D ['Vb']
################################# Dynamic
Data_1A_D= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\50%Dynamic-1A.xlsx')
print(Data_1A_D.head())

Time = Data_1A_D ['Time']

Ia_1_D = Data_1A_D ['Ia']
Ib_1_D = Data_1A_D ['Ib']
If_1_D = Data_1A_D ['If']
Va_1_D = Data_1A_D ['Va']
Vb_1_D = Data_1A_D ['Vb']
################################# Dynamic
Data_2A_D= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\50%Dynamic-2A.xlsx')
print(Data_2A_D.head())

Time = Data_2A_D ['Time']

Ia_2_D = Data_2A_D ['Ia']
Ib_2_D = Data_2A_D ['Ib']
If_2_D = Data_2A_D ['If']
Va_2_D = Data_2A_D ['Va']
Vb_2_D = Data_2A_D ['Vb']

################################# Dynamic
Data_3A_D= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\50%Dynamic-3A.xlsx')
print(Data_3A_D.head())

Time = Data_3A_D ['Time']

Ia_3_D = Data_3A_D ['Ia']
Ib_3_D = Data_3A_D ['Ib']
If_3_D = Data_3A_D ['If']
Va_3_D = Data_3A_D ['Va']
Vb_3_D = Data_3A_D ['Vb']

################################# Dynamic
Data_4A_D= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\50%Dynamic-4A.xlsx')
print(Data_4A_D.head())

Time = Data_4A_D ['Time']

Ia_4_D = Data_4A_D ['Ia']
Ib_4_D = Data_4A_D ['Ib']
If_4_D = Data_4A_D ['If']
Va_4_D = Data_4A_D ['Va']
Vb_4_D = Data_4A_D ['Vb']


################################# Dynamic
Data_5A_D= pd.read_excel(r'C:\Users\karami\Desktop\data_tir_1401\Telegram\50%Dynamic-5A.xlsx')
print(Data_5A_D.head())

Time = Data_5A_D ['Time']

Ia_5_D = Data_5A_D ['Ia']
Ib_5_D = Data_5A_D ['Ib']
If_5_D = Data_5A_D ['If']
Va_5_D = Data_5A_D ['Va']
Vb_5_D = Data_5A_D ['Vb']




data = pd.DataFrame({
                     
                     'Time': Time,
                     'He_Ia0A' : Ia_0_H,
                     'He_Ib0A' : Ib_0_H,
                     'He_If0A' : If_0_H,
                     'He_Va0A' : Va_0_H,
                     'He_Vb0A' : Vb_0_H,

                     'He_Ia1A' : Ia_1_H,
                     'He_Ib1A' : Ib_1_H,
                     'He_If1A' : If_1_H,
                     'He_Va1A' : Va_1_H,
                     'He_Vb1A' : Vb_1_H,

    
                     'He_Ia2A' : Ia_2_H,
                     'He_Ib2A' : Ib_2_H,
                     'He_If2A' : If_2_H,
                     'He_Va2A' : Va_2_H,
                     'He_Vb2A' : Vb_2_H,

    
                     'He_Ia3A' : Ia_3_H,
                     'He_Ib3A' : Ib_3_H,
                     'He_If3A' : If_3_H,
                     'He_Va3A' : Va_3_H,
                     'He_Vb3A' : Vb_3_H,

                     'He_Ia4A' : Ia_4_H,
                     'He_Ib4A' : Ib_4_H,
                     'He_If4A' : If_4_H,
                     'He_Va4A' : Va_4_H,
                     'He_Vb4A' : Vb_4_H,

                     'He_Ia5A' : Ia_5_H,
                     'He_Ib5A' : Ib_5_H,
                     'He_If5A' : If_5_H,
                     'He_Va5A' : Va_5_H,
                     'He_Vb5A' : Vb_5_H,

                     
                     'Fa_M_Ia0A' : Ia_0_M,
                     'Fa_M_Ib0A' : Ib_0_M,
                     'Fa_M_If0A' : If_0_M,
                     'Fa_M_Va0A' : Va_0_M,
                     'Fa_M_Vb0A' : Vb_0_M,

                     'Fa_M_Ia1A' : Ia_1_M,
                     'Fa_M_Ib1A' : Ib_1_M,
                     'Fa_M_If1A' : If_1_M,
                     'Fa_M_Va1A' : Va_1_M,
                     'Fa_M_Vb1A' : Vb_1_M,

                     'Fa_M_Ia2A' : Ia_2_M,
                     'Fa_M_Ib2A' : Ib_2_M,
                     'Fa_M_If2A' : If_2_M,
                     'Fa_M_Va2A' : Va_2_M,
                     'Fa_M_Vb2A' : Vb_2_M,

                     'Fa_M_Ia3A' : Ia_3_M,
                     'Fa_M_Ib3A' : Ib_3_M,
                     'Fa_M_If3A' : If_3_M,
                     'Fa_M_Va3A' : Va_3_M,
                     'Fa_M_Vb3A' : Vb_3_M,

                     'Fa_M_Ia4A' : Ia_4_M,
                     'Fa_M_Ib4A' : Ib_4_M,
                     'Fa_M_If4A' : If_4_M,
                     'Fa_M_Va4A' : Va_4_M,
                     'Fa_M_Vb4A' : Vb_4_M,

                     'Fa_M_Ia5A' : Ia_5_M,
                     'Fa_M_Ib5A' : Ib_5_M,
                     'Fa_M_If5A' : If_5_M,
                     'Fa_M_Va5A' : Va_5_M,
                     'Fa_M_Vb5A' : Vb_5_M,

                                                                             
                     'Fa_D_Ia0A' : Ia_0_D,
                     'Fa_D_Ib0A' : Ib_0_D,
                     'Fa_D_If0A' : If_0_D,
                     'Fa_D_Va0A' : Va_0_D,
                     'Fa_D_Vb0A' : Vb_0_D,

                     'Fa_D_Ia1A' : Ia_1_D,
                     'Fa_D_Ib1A' : Ib_1_D,
                     'Fa_D_If1A' : If_1_D,
                     'Fa_D_Va1A' : Va_1_D,
                     'Fa_D_Vb1A' : Vb_1_D,

                     'Fa_D_Ia2A' : Ia_2_D,
                     'Fa_D_Ib2A' : Ib_2_D,
                     'Fa_D_If2A' : If_2_D,
                     'Fa_D_Va2A' : Va_2_D,
                     'Fa_D_Vb2A' : Vb_2_D,

                     'Fa_D_Ia3A' : Ia_3_D,
                     'Fa_D_Ib3A' : Ib_3_D,
                     'Fa_D_If3A' : If_3_D,
                     'Fa_D_Va3A' : Va_3_D,
                     'Fa_D_Vb3A' : Vb_3_D,

                     'Fa_D_Ia4A' : Ia_4_D,
                     'Fa_D_Ib4A' : Ib_4_D,
                     'Fa_D_If4A' : If_4_D,
                     'Fa_D_Va4A' : Va_4_D,
                     'Fa_D_Vb4A' : Vb_4_D,

                     'Fa_D_Ia5A' : Ia_5_D,
                     'Fa_D_Ib5A' : Ib_5_D,
                     'Fa_D_If5A' : If_5_D,
                     'Fa_D_Va5A' : Va_5_D,
                      'Fa_D_Vb5A' : Vb_5_D

                     
                     })
print(data.head())





def FFT(x):
    """
    A recursive implementation of 
    the 1D Cooley-Tukey FFT, the 
    input should have a length of 
    power of 2. 
    """
    N = len(x)
    
    if N == 1:
        return x
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = \
          np.exp(-2j*np.pi*np.arange(N)/ N)
        
        X = np.concatenate(\
            [X_even+factor[:int(N/2)]*X_odd,
             X_even+factor[int(N/2):]*X_odd])
        return X


def get_tensors_from_xlsx(files_name):  
  acquisitions = {}
  for key in Gen_xkhz():
      print(len(key))
      file_name = files_name[key]
      My_file = data[file_name]
      acquisitions[key] = My_file.values.reshape(1,-1)[0]
  return acquisitions




def Gen_segmentation(acquisitions, sample_size=512, max_samples=None):
  '''
  Segments the acquisitions.
  sample_size is the size of each segment.
  max_samples is used for debug purpouses and 
  reduces the number of samples from each acquisition.
  '''
  origin = []
  data = np.empty((0,sample_size,1))
  n = len(acquisitions)
  for i,key in enumerate(acquisitions):
    acquisition_size = len(acquisitions[key])
    n_samples = acquisition_size//sample_size
    if max_samples is not None and max_samples > 0 and n_samples > max_samples:
      n_samples = max_samples
    print('{}/{} --- {}: {}'.format(i+1, n, key, n_samples))
    origin.extend([key for _ in range(n_samples)])
    data = np.concatenate((data,
           acquisitions[key][:(n_samples*sample_size)].reshape(
               (n_samples,sample_size,1))))
  return data,origin




def select_samples(regex, X, y):
  '''
  Selects samples wich has some regex pattern in its name.
  '''
  mask = [re.search(regex,label) is not None for label in y]
  return X[mask],y[mask]

def join_labels(regex, y):
  '''
  Excludes some regex patterns from the labels, 
  making some samples to have the same label.
  '''
  return np.array([re.sub(regex, '', label) for label in y])


def get_groups(regex, y):
  '''
  Generates a list of groups of samples with 
  the same regex patten in its label.
  '''
  groups = list(range(len(y)))
  for i,label in enumerate(y):
    match = re.search(regex,label)
    groups[i] = match.group(0) if match else None
  return groups





files_name = Gen_xkhz()

acquisitions = get_tensors_from_xlsx(files_name)

a=acquisitions['He_Ia1A'];

acquisitions.get('He_Ia0A').shape

print(acquisitions)
list(acquisitions.keys())







mm=data['Fa_D_Vb5A']
print(acquisitions['Fa_D_Vb5A'])

# sampling rate
sr = 40000
X=FFT(acquisitions['Fa_D_Vb5A'])

# calculate the frequency
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T 

plt.figure(figsize = (12, 6))
plt.subplot(121)
plt.stem(freq, abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')

# Get the one-sided specturm
n_oneside = N//2
# get the one side frequency
f_oneside = freq[:n_oneside]

# normalize the amplitude
X_oneside =X[:n_oneside]/n_oneside

plt.subplot(122)
plt.stem(f_oneside, abs(X_oneside), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('Normalized FFT Amplitude |X(freq)|')
plt.tight_layout()
plt.show()






















signal_data,signal_origin = Gen_segmentation(acquisitions, 40)

signal_data.shape



def samples_relabel(regex, rp, y):
  '''
  Selects samples wich has some regex pattern in its name.
  '''
  mask = [re.search(regex,label) is not None for label in y]
  y[mask]=rp
  return y

relabel_signal = np.array(signal_origin)



regex = '^(He).*'
rp = 'He'
relabel_signal = samples_relabel(regex, rp, relabel_signal)



regex = '^(Fa_M).*'
rp = 'Fa_M'
relabel_signal = samples_relabel(regex, rp, relabel_signal)



regex = '^(Fa_D).*'
rp = 'Fa_D'
relabel_signal = samples_relabel(regex, rp, relabel_signal)

np.where(relabel_signal == "He")


samples = '^(He)|(Fa).*'


# In[ ]:


X,y = select_samples(samples, signal_data, relabel_signal)


# In[ ]:


print(len(set(y)),set(y))


# In[ ]:


X[158].shape


# In[ ]:



plt.figure(figsize=(8,3))
plt.ylim([-7,7])
plt.plot(X[8000])


# In[ ]:


labels = pd.Categorical(y, categories = set(y)).codes
labels


# In[ ]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.15, shuffle=True)

nsamples, nx, ny = X_train.shape
d2_train_dataset = X_train.reshape((nsamples,nx*ny))
nsamples2, nx, ny = X_test.shape
d3_test_dataset = X_test.reshape((nsamples2,nx*ny))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.svm import SVC
net_SVC = SVC(C=1.0, kernel='rbf', gamma='auto')
net_SVC.fit(d2_train_dataset,y_train)
print("net_SVC accuracy is {} on Train Dataset".format(net_SVC.score(d2_train_dataset,y_train)))
print("net_SVC accuracy is {} on Test Dataset".format(net_SVC.score(d3_test_dataset,y_test)))

predictions = net_SVC.predict(d3_test_dataset)
cm = confusion_matrix(y_test, predictions, labels=net_SVC.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=net_SVC.classes_)
disp.plot()
plt.show()

from sklearn.svm import LinearSVC
lin_svc = LinearSVC()
lin_svc.fit(d2_train_dataset,y_train)
print("lin_svc accuracy is {} on Train Dataset".format(lin_svc.score(d2_train_dataset,y_train)))
print("lin_svc accuracy is {} on Test Dataset".format(lin_svc.score(d3_test_dataset,y_test)))


from sklearn.svm import NuSVC
net_NuSVC = NuSVC()
net_NuSVC.fit(d2_train_dataset,y_train)
print("net_NuSVC accuracy is {} on Train Dataset".format(net_NuSVC.score(d2_train_dataset,y_train)))
print("net_NuSVC accuracy is {} on Test Dataset".format(net_NuSVC.score(d3_test_dataset,y_test)))


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge




from sklearn.tree import DecisionTreeRegressor

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_3 = DecisionTreeRegressor(max_depth=8)
regr_4 = DecisionTreeRegressor(max_depth=25)

regr_1.fit(d2_train_dataset, y_train)
regr_2.fit(d2_train_dataset, y_train)
regr_3.fit(d2_train_dataset, y_train)
regr_4.fit(d2_train_dataset, y_train)
# Predict

y_1 = regr_1.predict(d3_test_dataset)
y_2 = regr_2.predict(d3_test_dataset)
y_3 = regr_3.predict(d3_test_dataset)
y_4 = regr_4.predict(d3_test_dataset)

# Make predictions using the testing set
print("DecisionTreeRegressor_regr_1 accuracy is {} on Train Dataset".format(regr_1.score(d2_train_dataset,y_train)))
print("DecisionTreeRegressor_regr_1 accuracy is {} on Test Dataset".format(regr_1.score(d3_test_dataset,y_test)))

# Make predictions using the testing set
print("DecisionTreeRegressor_regr_2 accuracy is {} on Train Dataset".format(regr_2.score(d2_train_dataset,y_train)))
print("DecisionTreeRegressor_regr_2 accuracy is {} on Test Dataset".format(regr_2.score(d3_test_dataset,y_test)))

# Make predictions using the testing set
print("DecisionTreeRegressor_regr_3 accuracy is {} on Train Dataset".format(regr_3.score(d2_train_dataset,y_train)))
print("DecisionTreeRegressor_regr_3 accuracy is {} on Test Dataset".format(regr_3.score(d3_test_dataset,y_test)))


print("DecisionTreeRegressor_regr_4 accuracy is {} on Train Dataset".format(regr_4.score(d2_train_dataset,y_train)))
print("DecisionTreeRegressor_regr_4 accuracy is {} on Test Dataset".format(regr_4.score(d3_test_dataset,y_test)))



# Create linear regression object
Myregression = LinearRegression()
# Train the model using the training sets
Myregression.fit(d2_train_dataset, y_train)

# Make predictions using the testing set
y_test = Myregression.predict(d3_test_dataset)

print("LinearRegression Accuracy on Train Data: {}".format(Myregression.score(d2_train_dataset,y_train)))
print("LinearRegression Accuracy on Test Data: {}".format(Myregression.score(d3_test_dataset,y_test)))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
# Train the model using the training sets
gnb.fit(d2_train_dataset, y_train)

# Make predictions using the testing set
y_test = gnb.predict(d3_test_dataset)

print("Naive_bayes accuracy is {} on Train Dataset".format(gnb.score(d2_train_dataset,y_train)))
print("Naive_bayes accuracy is {} on Test Dataset".format(gnb.score(d3_test_dataset,y_test)))








knn = KNeighborsClassifier(n_neighbors = 1) #n_neighbors = k
knn.fit(d2_train_dataset,y_train)
print("k={}NN Accuracy on Train Data: {}".format(knn.score(d2_train_dataset,y_train)))
print("k={}NN Accuracy on Test Data: {}".format(knn.score(d3_test_dataset,y_test)))




early_stop = EarlyStopping(monitor='loss', patience=2)
model = Sequential()

model.add(Dense(40, activation='relu', input_shape=(40,),kernel_initializer='random_uniform'))

model.add(Dense(1280, activation='relu',kernel_initializer='random_uniform'))
          
model.add(Dense(512, activation='relu',kernel_initializer='random_uniform'))

model.add(Dropout(0.98))

model.add(Dense(256, activation='relu',kernel_initializer='random_uniform'))

# model.add(Dense(128, activation='relu',kernel_initializer='random_uniform'))

# model.add(Dense(64, activation='relu',kernel_initializer='random_uniform'))

model.add(Dense(3, activation='softmax',kernel_initializer='random_uniform'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

y = LabelEncoder().fit_transform(y)



# In[ ]:


hist = model.fit(d2_train_dataset , y_train, batch_size=100, epochs=4500,validation_split=0.2)


# In[ ]:


plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
