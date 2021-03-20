# ProAct



## status
> Development Stage
<!--
#Put all data in data folder - each dataset has to be in the form:

Name: Sequence: Target

rename to

ProTSAR
ProtSAR  
ProPhySAR
ProDescSAR
ActSeqR


#downloading and importing PyBioMed:
 curl -LJO https://github.com/gadsbyfly/PyBioMed/archive/master.zip
 unzip PyBioMed-master.zip
 rm -r PyBioMed-master.zip
 mv PyBioMed-master PyBioMed



Import Dataset -> Calculate AAI for all sequences -> Calcualte descriptors from dataset ->
Build predictive modles -> Output Results


#check using standardscaler correctly;

from numpy import asarray
from sklearn.preprocessing import StandardScaler
# define data
data = asarray([[100, 0.001],
				[8, 0.05],
				[50, 0.005],
				[88, 0.07],
				[4, 0.1]])
print(data)
# define standard scaler
scaler = StandardScaler()
# transform data
scaled = scaler.fit_transform(data)
print(scaled) -->
