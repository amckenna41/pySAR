import itertools
import pandas as pd
import numpy as np
import itertools
from difflib import get_close_matches
from sklearn.preprocessing import StandardScaler
from PyBioMed.PyBioMed.PyProtein import AAComposition

scaler = StandardScaler()
#
# class Descriptors():
#
#     all_desc = ['aa_comp', 'dipeptide_comp', 'tripeptide_comp', 'norm_moreaubroto_autocorrelation','moran_autocorrelation','geary_autocorrelation',
#                'composition','transition','distribution', 'conjoint_triad','seq_order_coupling_number','quasi_seq_order_descriptors','pseudo_aa_comp', 'amphipilic_pseudo_aa_comp']
#
#     def __init__(self, descriptors_file):
#
#         self.descriptors_file = descriptors_file
#         #read descriptor df
#         try:
#             self.descriptors = pd.read_csv(self.descriptors_file)
#         except:
#             print('Erro opening file')
#
#         self.descriptors.drop(self.descriptors.columns[[0,1]], axis = 1, inplace=True)
#
#         #read descriptors csv file
#
#     def get_descriptor(self, descriptor):
#
#         assert descriptor in all_desc, 'Descriptor param must be in: {} '.format(all_desc)
#
#         if descriptor == 'aa_comp':
#              desc_df = self.descriptors.iloc[:,: 20]
#         elif descriptor == 'dipeptide_comp':
#              desc_df = self.descriptors.iloc[:,20:420]
#         elif descriptor == 'tripeptide_comp':
#              desc_df = self.descriptors.iloc[:,420:8420]
#         elif descriptor == 'norm_moreaubroto_autocorrelation':
#              desc_df = self.descriptors.iloc[:,8420:8660]
#         elif descriptor == 'moran_autocorrelation':
#              desc_df = self.descriptors.iloc[:,8660: 8900]
#         elif descriptor == 'geary_autocorrelation':
#              desc_df = self.descriptors.iloc[:,8900:9140]
#         elif descriptor == 'composition':
#              desc_df = self.descriptors.iloc[:,9140:9161]
#         elif descriptor == 'transition':
#              desc_df = self.descriptors.iloc[:,9161:9182]
#         elif descriptor == 'distribution':
#              desc_df = self.descriptors.iloc[:,9182:9287]
#         elif descriptor == 'conjoint_triad':
#              desc_df = self.descriptors.iloc[:,9287:9630]
#         elif descriptor == 'seq_order_coupling_number':
#              desc_df = self.descriptors.iloc[:,9630:9690]
#         elif descriptor == 'quasi_seq_order_descriptors':
#              desc_df = self.descriptors.iloc[:,9690:9790]
#         elif descriptor == 'pseudo_aa_comp':
#              desc_df = self.descriptors.iloc[:,9790:9840]
#         elif descriptor == 'amphipilic_pseudo_aa_comp':
#              desc_df = self.descriptors.iloc[:,9840:9920]
#         elif descriptor == 'power':
#              desc_df = self.descriptors.iloc[:,9920:9921]
#         elif descriptor == 'real':
#              desc_df = self.descriptors.iloc[:,9921:9922]
#         elif descriptor == 'imag':
#              desc_df = self.descriptors.iloc[:,9922:9923]
#
#         return desc_df
#
#         def get_all_descriptors(self):
#
#             return self.descriptors
#
#         def __shape__(self):
#
#             return self.descriptors.shape
#
#         def descriptor_shape(self, descriptor):
#
#             return descriptor.shape
#
#         #return desc from descriptors
#


#building descriptors usinf PyBioMed module
class Descriptors():

    def __init__(self, protein_seqs):

        self.protein_seqs = protein_seqs

        # self.validModels = self.valid_models()
        #
        # modelMatches = get_close_matches(self.algorithm,self.validModels)
        #
        # if modelMatches!=[]:
        #     self.algorithm = modelMatches[0]
        # else:
        #     raise ValueError('Input algorithm ('+ self.algorithm + ') not in available models /n '+self.validModels)

    def aa_composition(self):

        AA_comp = []
        encoded_AA = []
        for seq in self.protein_seqs:
            AAC=AAComposition.CalculateAAComposition(seq)
            AA_comp.append(AAC)

        for d in range(0,len(AA_comp)):
            encoded_AA.append(list(AA_comp[d].values()))

        encoded_AA = (list(itertools.chain.from_iterable(encoded_AA)))
        encoded_AA = np.reshape(encoded_AA, (len(AA_comp), len(AA_comp[0])))

        encoded_AA = scaler.fit_transform(encoded_AA)

        return encoded_AA

    def dipeptide_composition(self):

        dipeptide_comp = []

        for seq in self.protein_seqs:
            AADipeptide=AAComposition.CalculateDipeptideComposition(seq)
            dipeptide_comp.append(AADipeptide)

        return dipeptide_comp

    def tripeptide_composition(self):

        tripeptide_comp = []

        for seq in self.protein_seqs:
            AATripeptide=AAComposition.GetSpectrumDict()
            tripeptide_comp.append(AATripeptide)

        return tripeptide_comp

    def get_descriptor_encoding(self,descriptor):

        self.validDesc = self.valid_descriptors()
        descMatches = get_close_matches(descriptor,self.validDesc)

        if descMatches!=[]:
            desc = descMatches[0]
        else:
            raise ValueError('Input descriptor ('+ desc + ') not in available descriptors /n '+self.valid_descriptors)

        if desc == 'aa_composition':
            desc_encoding = self.aa_composition()

        return desc_encoding

    def valid_descriptors(self):

        validDesc = ['aa_composition','dipeptide_composition','tripeptide_composition']

        return validDesc

#
#
#
# all_descriptors_combo_2 = list(itertools.combinations(all_descriptors, 2))
# # all_descriptors_combo_3 = list(itertools.combinations(all_descriptors, 3))
#
# #desc_combo_3 = pd.DataFrame(columns=['Descriptors', 'R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Var'])
# desc_combo_2 = pd.DataFrame(columns=['Descriptors', 'R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Var'])
#
# temp_desc_metrics_df = desc_combo_2.copy()
#
# def tune_descriptor_combo_2(model, descriptor_df, Y):
#
#     print('Building  model using {} with two descriptors for features - no aaindex encoding....\n\n'.format(type(model).__name__))
#
#     desc_ = []
#     r2_ = []
#     mse_ = []
#     rmse_ = []
#     rpd_ = []
#     mae_ = []
#     explained_var_ = []
#     msle_ = []
#     desc_count = 1
#
#     for desc in all_descriptors_combo_2:
#
#
#         print('Descriptors {} ###### {}/{}'.format(desc , desc_count, len(all_descriptors_combo_2)))
#         desc_count+=1
#
#         X1 = get_descriptor(descriptor_df, desc[0])
#         X2 = get_descriptor(descriptor_df, desc[1])
#         X_ = [X1, X2]
#         X = pd.concat(X_, axis=1)
#
#         X = scaler.fit_transform(X)
#         X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#
#         model_copy = clone(model)
#
#         # model_copy = PLSRegression()
#
#         model_copy.fit(X_train, Y_train)
#         Y_pred = model_copy.predict(X_test)
#
#         r2 = model_copy.score(X_train, Y_train)
#         rmse, mse, mae, rpd, explained_var, msle = eval_metrics(Y_test, Y_pred)
#
#         desc_.append(desc)
#         r2_.append(r2)
#         rmse_.append(rmse)
#         mse_.append(mse)
#         rpd_.append(rpd)
#         mae_.append(mae)
#         explained_var_.append(explained_var)
#         msle_.append(msle)
#
#     desc_combo_2['Descriptors'] = desc_
#     desc_combo_2['R2'] = r2_
#     desc_combo_2['RMSE'] = rmse_
#     desc_combo_2['MSE'] = mse_
#     desc_combo_2['RPD'] = rpd_
#     desc_combo_2['MAE'] = mae_
#     desc_combo_2['Explained Var'] = explained_var_
#     desc_combo_2['MSLE'] = msle_
#     # desc_metrics_df.sort_values(by=['R2'], inplace=True, ascending=False)
#
#     X = desc_combo_2
#
#     mse_df = X.loc[:, X.columns.str.startswith('MSE')]
#     rmse_df = X.loc[:, X.columns.str.startswith('RMSE')]
#     r2_df =  X.loc[:, X.columns.str.startswith('R2')]
#     rpd_df = X.loc[:, X.columns.str.startswith('RPD')]
#     mae_df =  X.loc[:, X.columns.str.startswith('MAE')]
#     explainedVar_df = X.loc[:, X.columns.str.startswith('Explained')]
#     msle_df = X.loc[:, X.columns.str.startswith('MSLE')]
#
#     mse_df.insert(0, 'Descriptors', X['Descriptors'])
#     rmse_df.insert(0, 'Descriptors', X['Descriptors'])
#     r2_df.insert(0, 'Descriptors', X['Descriptors'])
#     mae_df.insert(0, 'Descriptors', X['Descriptors'])
#     rpd_df.insert(0, 'Descriptors', X['Descriptors'])
#     explainedVar_df.insert(0, 'Descriptors', X['Descriptors'])
#     msle_df.insert(0, 'Descriptors', X['Descriptors'])
#
#
#     return X, mse_df, rmse_df, r2_df, mae_df, rpd_df, explainedVar_df, msle_df
