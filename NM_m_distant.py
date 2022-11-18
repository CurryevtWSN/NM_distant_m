import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
#应用标题
st.set_page_config(page_title='The Establishment of the Metastasis and Prognosis Model for Patients with Nodular Melanoma Incorporating Machine Learning Algorithms ')
st.title('The Establishment of the Metastasis and Prognosis Model for Patients with Nodular Melanoma Incorporating Machine Learning Algorithms ')
st.sidebar.markdown('## Variables')
Marital = st.sidebar.selectbox('Marital',('Married','Unmarried','Unknown'),index=1)
Gender = st.sidebar.selectbox('Gender',('Male','Female'),index=1)
Primary_Site = st.sidebar.selectbox('Primary_Site',('Skin of trunk','Skin of upper limb and shoulder','Skin of lower limb and hip',
                                                    'Skin of other and unspecified parts of face','External ear','Eyelid','Vulva','Lip',
                                                    'Skin of scalp and neck','Other'),index=1)
Laterality = st.sidebar.selectbox('Laterality',('Left','Right','Paired site, midline tumor','Not a paired site',
                                                'Only one side, side unspecified','Bilateral, single primary','Unknown'),index=1)
Surgery = st.sidebar.selectbox('Surgery',('No surgery of primary site','Local tumor destruction','Biopsy of primary tumor followed by a gross excision of the lesion, does not have to be done under the same anesthesia',
                                          'Wide excision or reexcision of lesion or local amputation with margins more than 1 cm. Margins MUST be microscopically negative',
                                          'Other/Unknown'),index=1)
Radiation = st.sidebar.selectbox('Radiation',('None/Unknown','Yes'),index=1)
Chemotherapy = st.sidebar.selectbox('Chemotherapy',('None/Unknown','Yes'),index=1)
System_management = st.sidebar.selectbox('System_management',('None/Unknown','Yes'),index=1)
T = st.sidebar.selectbox('T',('T1','T2','T3','T4','TX'),index=1)
N = st.sidebar.selectbox('N',('No regional lymph node metastasis was observed','With regional lymph node metastasis'),index=1)


#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'Married':0,'Unmarried':1,'Unknown':2,'Male':0,'Female':1,'Skin of trunk':0,'Skin of upper limb and shoulder':1,'Skin of lower limb and hip':2,
       'Skin of other and unspecified parts of face':3,'External ear':4,'Eyelid':5,'Vulva':6,'Lip':7,'Skin of scalp and neck':8,'Other':9,
       'Left':0,'Right':1,'Paired site, midline tumor':3,'Not a paired site':4,'Only one side, side unspecified':5,'Bilateral, single primary':6,'Unknown':7,
       'No surgery of primary site':0,'Local tumor destruction':1,'Biopsy of primary tumor followed by a gross excision of the lesion, does not have to be done under the same anesthesia':2,
       'Wide excision or reexcision of lesion or local amputation with margins more than 1 cm. Margins MUST be microscopically negative':3,
       'Other/Unknown':4,'None/Unknown':0,'Yes':1,'T1':1,'T2':2,'T3':3,'T4':4,"TX":5,'No regional lymph node metastasis was observed':0,
       'With regional lymph node metastasis':1}
Marital =map[Marital]
Gender = map[Gender]
Primary_Site = map[Primary_Site]
Laterality =map[Laterality]
Surgery =map[Surgery]
Radiation =map[Radiation]
Chemotherapy =map[Chemotherapy]
System_management =map[System_management]
T =map[T]
N = map[N]

# 数据读取，特征标注
hp_train = pd.read_csv('E:\\Spyder_2022.3.29\\output\\machinel\\lwl_output\\NM_WSN\\nmdata.csv')
hp_train['M'] = hp_train['M'].apply(lambda x : +1 if x==1 else 0)
features =["Marital","Gender","Primary_Site","Laterality",'Surgery','Radiation','Chemotherapy','System_management','T','N']
target = 'M'
random_state_new = 50
data = hp_train[features]
X_data = data
X_ros = np.array(X_data)
y_ros = np.array(hp_train[target])
oversample = SMOTE(random_state = random_state_new)
X_ros, y_ros = oversample.fit_resample(X_ros, y_ros)
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='lbfgs',
                    alpha=0.0001,
                    batch_size='auto',
                    learning_rate='constant',
                    learning_rate_init=0.01,
                    power_t=0.5,
                    max_iter=200,
                    shuffle=True, random_state=random_state_new)
XGB_model = mlp
XGB_model.fit(X_ros, y_ros)
sp = 0.5
#figure
is_t = (XGB_model.predict_proba(np.array([[Marital,Gender,Primary_Site,Laterality,Surgery,Radiation,Chemotherapy,System_management,T,N]]))[0][1])> sp
prob = (XGB_model.predict_proba(np.array([[Marital,Gender,Primary_Site,Laterality,Surgery,Radiation,Chemotherapy,System_management,T,N]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk distant metastasis'
else:
    result = 'Low Risk distant metastasis'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk distant metastasis':
        st.balloons()
    st.markdown('## Probability of High Risk distant metastasis group:  '+str(prob)+'%')
