######################
# Import libraries
######################
from matplotlib import cm
from rdkit.Chem.Draw import SimilarityMaps
from numpy import loadtxt
from queue import Empty
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import PandasTools
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.model_selection import permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import pairwise_distances
import joblib
from IPython.display import HTML
import matplotlib.pyplot as plt
from stmol import showmol
import py3Dmol

######################
# Page Title
######################
st.write("<h1 style='text-align: center; color: blue;'> HDAC1 PREDICTOR v.1.0</h1>", unsafe_allow_html=True)
image = Image.open('app_logo.jpg')
st.image(image, use_column_width=True)
st.write("<h3 style='text-align: center; color: black;'> A machine learning Web application to assess the potential of histone deacetylase 1 (HDAC1) inhibitors.</h1>", unsafe_allow_html=True)
if st.button('App Characteristics'):
    st.write('The HDAC1  Predictor application provides an alternative method for assessing the potential of chemicals to be Histone deacetylas 1 (HDAC1) inhibitors.  Compound is classified as active if the predicted IC50 value is  lower than mean IC50 value of the reference drug Vorinostat (11.08 nM)  otherwise compound is  labeled as inactive. HDAC1 Detective makes predictions based on Quantitative Structure-Activity Relationship (QSAR) models build on curated datasets generated from ChEMBL database (ID: CHEMBL325). The consensus models were developed using open-source chemical descriptors based on ECFP4-like Morgan fingerprints and 2D RDKit descriptors, along with the random forest (RF), gradient boosting (GBM), support vector machines (SVM)  algorithms, using Python 3.7. The models were generated applying the best practices for QSAR model development and validation widely accepted by the community. Batch processing is available through https://github.com/ovttiras/HDAC1-inhibitors. For more information, please refer to our paper:')



# Select and read  saved model
models_option = st.sidebar.selectbox('Select consensus QSAR model for prediction', ('ECFP4', 'RDKit'))
     
if models_option == 'ECFP4':
    load_model_RF = pickle.load(open('FP/HDAC1_RF_ECFP4.pkl', 'rb'))
    load_model_GBM = pickle.load(open('FP/HDAC1_GBM_ECFP4.pkl', 'rb'))
    scale = joblib.load('FP/HDAC1_ws_for SVM.pkl')
    load_model_SVM = pickle.load(open('FP/HDAC1_SVM_ECFP4.pkl', 'rb'))
    x_tr = loadtxt('FP/x_tr.csv', delimiter=',')
    model_AD_limit = 4.11
    st.sidebar.header('Select the compounds to be predicted')
        # Read SMILES input
    SMILES = st.sidebar.checkbox('SMILES input')
    if SMILES:
        SMILES_input = ""
        compound_smiles = st.sidebar.text_area("Enter SMILES", SMILES_input)
        m = Chem.MolFromSmiles(compound_smiles)

        im = Draw.MolToImage(m)
        st.sidebar.image(im)
        
        if st.sidebar.button('PREDICT COMPOUND FROM SMILES'):
            # Calculate molecular descriptors
            f_vs = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024, useFeatures=False, useChirality=False)]
            def rdkit_numpy_convert(f_vs):
                output = []
                for f in f_vs:
                    arr = np.zeros((1,))
                    DataStructs.ConvertToNumpyArray(f, arr)
                    output.append(arr)
                    return np.asarray(output)


            X = rdkit_numpy_convert(f_vs)

            ######################
            # Pre-built model
            ######################

            X_s = scale.transform(X)

            # Apply model to make predictions
            prediction_RF = load_model_RF.predict(X)
            prediction_GBM = load_model_GBM.predict(X)
            prediction_SVM = load_model_SVM.predict(X)
            pred_consensus = 1 * \
            (((prediction_RF + prediction_GBM + prediction_SVM) / 3) >= 0.5)
            pred_consensus = np.array(pred_consensus)
            pred_consensus = np.where(pred_consensus == 1, "Active", "Inactive")


            # Estimination AD

            # load numpy array from csv file
            x_tr = loadtxt('FP/x_tr.csv', delimiter=',')
            model_AD_limit = 4.11
            neighbors_k_vs = pairwise_distances(x_tr, Y=X, n_jobs=-1)
            neighbors_k_vs.sort(0)
            similarity_vs = neighbors_k_vs
            cpd_value_vs = similarity_vs[0, :]
            cpd_AD_vs = np.where(cpd_value_vs <= model_AD_limit, "Inside AD", "Outside AD")

            st.header('**Prediction results:**')
            st.write('**HDAC1**: ', pred_consensus[0])
            st.write('**Applicability domain (AD)**: ', cpd_AD_vs[0])

            # Generate maps of fragment contribution
            
            def getProba(fp, predictionFunction):
                return predictionFunction((fp,))[0][1]


            def fpFunction(m, atomId=-1):
                fp = SimilarityMaps.GetMorganFingerprint(m,
                                                        atomId=atomId,
                                                        radius=2,
                                                        nBits=1024)
                return fp


            fig, maxweight = SimilarityMaps.GetSimilarityMapForModel(
                m, fpFunction, lambda x: getProba(x, load_model_RF.predict_proba), colorMap=cm.PiYG_r)
            st.write('**Predicted fragments contribution:**')
            st.pyplot(fig)
            st.write('The chemical fragments are colored in green (predicted to reduce inhibitory activity) or magenta (predicted to increase activity HDAC1 inhibitors). The gray isolines separate positive and negative contributions.')
            
            # 3D structure
            st.header('**3D structure of the studied compound:**')
            def makeblock(smi):
                mol = Chem.MolFromSmiles(smi)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                mblock = Chem.MolToMolBlock(mol)
                return mblock

            def render_mol(xyz):
                xyzview = py3Dmol.view()#(width=400,height=400)
                xyzview.addModel(xyz,'mol')
                xyzview.setStyle({'stick':{}})
                xyzview.setBackgroundColor('black')
                xyzview.zoomTo()
                showmol(xyzview,height=500,width=500)
            blk=makeblock(compound_smiles)
            render_mol(blk)
            st.write('You can use the scroll wheel on your mouse to zoom in or out a 3D structure of compound')
                    

    # Read SDF file 
    LOAD = st.sidebar.checkbox('Upload compounds from file sdf')
    if LOAD:
        uploaded_file = st.sidebar.file_uploader("Choose a file")
        if uploaded_file is not None:
            sdfInfo = dict(smilesName='SMILES', molColName='ROMol')
            
            moldf = PandasTools.LoadSDF(uploaded_file, **sdfInfo)
            st.header('**CHECKING STRUCTURES:**')
            st.write('Original data: ', len(moldf), 'molecules')
            # Rename ROMol
            moldf = moldf.rename(columns={'ROMol': 'Mol'})
            # Remove missing RDKit molecules
            moldf = moldf[pd.notnull(moldf['Mol'])]
            if 'StandardizerResult' in moldf.columns:
                moldf = moldf.drop(columns='StandardizerResult')
            # Columns
            st.write('Kept data: ', len(moldf), 'molecules')
            from molvs.validate import Validator
            fmt = '%(asctime)s - %(levelname)s - %(validation)s - %(message)s'
            validator = Validator(log_format=fmt)
            st.write('\n Problematic structures: \n', validator.validate(moldf))

        
            # Calculate molecular descriptors

            def calcfp(mol,funcFPInfo=dict(radius=2,nBits=1024,useFeatures=False,useChirality = False)):
                arr = np.zeros((1,))
                fp = GetMorganFingerprintAsBitVect(mol, **funcFPInfo)
                DataStructs.ConvertToNumpyArray(fp, arr)
                return arr

            moldf['Descriptors'] = moldf.Mol.apply(calcfp)
            X = np.array(list(moldf['Descriptors'])).astype(int)
                
            ######################
            # Pre-built model
            ######################

            X_s = scale.transform(X)

            # Apply model to make predictions
            prediction_RF = load_model_RF.predict(X)
            prediction_GBM = load_model_GBM.predict(X)
            prediction_SVM = load_model_SVM.predict(X)
            pred_consensus = 1 * \
            (((prediction_RF + prediction_GBM + prediction_SVM) / 3) >= 0.5)

            pred_consensus = np.array(pred_consensus)
            pred_consensus = np.where(pred_consensus == 1, "Active", "Inactive")
        

            # Estimination AD

            # load numpy array from csv file
            x_tr = loadtxt('FP/x_tr.csv', delimiter=',')
            model_AD_limit = 4.11
            neighbors_k_vs = pairwise_distances(x_tr, Y=X, n_jobs=-1)
            neighbors_k_vs.sort(0)
            similarity_vs = neighbors_k_vs
            cpd_value_vs = similarity_vs[0, :]
            cpd_AD_vs = np.where(cpd_value_vs <= model_AD_limit, "Inside AD", "Outside AD")



            # Generate maps of fragment contribution
            def getProba(fp, predictionFunction):
                return predictionFunction((fp,))[0][1]


            def fpFunction(m, atomId=-1):
                fp = SimilarityMaps.GetMorganFingerprint(m,
                                                        atomId=atomId,
                                                        radius=2,
                                                        nBits=1024)
                return fp
            #Print and download common results
            st.header('**RESULTS OF PREDICTION:**')
            if st.button('Show results as table'):
                moldf.drop(columns='Descriptors', inplace=True)
                moldf.drop(columns='Mol', inplace=True)
                moldf.drop(columns='ID', inplace=True)                
                pred_beta = pd.DataFrame({'HDAC1 activity': pred_consensus,'Applicability domain (AD)': cpd_AD_vs}, index=None)
                predictions = pd.concat([moldf, pred_beta], axis=1)
                st.dataframe(predictions)           
                def convert_df(df):
                    return df.to_csv().encode('utf-8')  
                csv = convert_df(predictions)

                st.download_button(
                    label="Download results of prediction as CSV",
                    data=csv,
                    file_name='Results.csv',
                    mime='text/csv',
                )

            # Print results for each molecules
            if st.button('Show results and map of fragments contribution for each molecule separately'):
                st.header('**Prediction results:**')
                for i in range(len(moldf.Mol)):
                    a= moldf['SMILES']
                    b=list(a)  
                for i in range(len(b)):
                    m = Chem.MolFromSmiles(b[i])
                    im = Draw.MolToImage(m)
                    st.write('**COMPOUNDS NUMBER **' + str(i+1) + '**:**')
                    st.write('**2D structure of compound number **' + str(i+1) + '**:**')
                    st.image(im)
                    # 3D structure
                    st.write('**3D structure of compound number **'+ str(i+1) + '**:**')
                    def makeblock(smi):
                        mol = Chem.MolFromSmiles(smi)
                        mol = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol)
                        mblock = Chem.MolToMolBlock(mol)
                        return mblock

                    def render_mol(xyz):
                        xyzview = py3Dmol.view()#(width=400,height=400)
                        xyzview.addModel(xyz,'mol')
                        xyzview.setStyle({'stick':{}})
                        xyzview.setBackgroundColor('black')
                        xyzview.zoomTo()
                        showmol(xyzview,height=500,width=500)
                    blk=makeblock(b[i])
                    render_mol(blk)
                    st.write('You can use the scroll wheel on your mouse to zoom in or out a 3D structure of compound')


                    st.write('**Smiles for compound number **'+ str(i+1) + '**:**', b[i])
                    st.write('**HDAC1:** ', pred_consensus[i])
                    st.write('**Applicability domain (AD):** ', cpd_AD_vs[i])
                    fig, maxweight = SimilarityMaps.GetSimilarityMapForModel(m, fpFunction, lambda x: getProba(x, load_model_RF.predict_proba), colorMap=cm.PiYG_r)
                    st.write('**Predicted fragments contribution for compound number **'+ str(i+1) + '**:**')
                    st.pyplot(fig)
                    st.write('The chemical fragments are colored in green (predicted to reduce inhibitory activity) or magenta (predicted to increase activity HDAC1 inhibitors). The gray isolines separate positive and negative contributions.')
                    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
            

if models_option == 'RDKit':
    load_model_RF = pickle.load(open('RDKit/HDAC1_RF_RDKit.pkl', 'rb'))
    load_model_GBM = pickle.load(open('RDKit/HDAC1_GBM_RDKit.pkl', 'rb'))
    scale = joblib.load('RDKit/HDAC1_ws_for SVM.pkl')
    load_model_SVM = pickle.load(open('RDKit/HDAC1_SVM_RDKit.pkl', 'rb'))
    st.sidebar.header('Select the compounds to be predicted')
        
        # Read SMILES input
    SMILES = st.sidebar.checkbox('SMILES input')
    if SMILES:
        SMILES_input = ""
        compound_smiles = st.sidebar.text_area("Enter SMILES", SMILES_input)
        m = Chem.MolFromSmiles(compound_smiles)

        im = Draw.MolToImage(m)
        st.sidebar.image(im)

        if st.sidebar.button('PREDICT COMPOUND FROM SMILES'):
            # Calculate molecular descriptors
            mols_ts = []
            mols_ts.append(m)
            descr_ts = []
            calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
            header = calc.GetDescriptorNames()
            for m in mols_ts:
                descr_ts.append(calc.CalcDescriptors(m))
            X = np.asarray(descr_ts)
            
            ######################
            # Pre-built model
            ######################

            X_s = scale.transform(X)

            # Apply model to make predictions
            prediction_RF = load_model_RF.predict(X)
            prediction_GBM = load_model_GBM.predict(X)
            prediction_SVM = load_model_SVM.predict(X)
            pred_consensus = 1 * \
            (((prediction_RF + prediction_GBM + prediction_SVM) / 3) >= 0.5)
            pred_consensus = np.array(pred_consensus)
            pred_consensus = np.where(pred_consensus == 1, "Active", "Inactive")


            # Estimination AD

            # load numpy array from csv file
            x_tr = loadtxt('RDKit/x_tr.csv', delimiter=',')
            model_AD_limit = 1290022998.45
            neighbors_k_vs = pairwise_distances(x_tr, Y=X, n_jobs=-1)
            neighbors_k_vs.sort(0)
            similarity_vs = neighbors_k_vs
            cpd_value_vs = similarity_vs[0, :]
            cpd_AD_vs = np.where(cpd_value_vs <= model_AD_limit, "Inside AD", "Outside AD")
            
            st.header('**Prediction results:**')
            st.write('**HDAC1**: ', pred_consensus[0])
            st.write('**Applicability domain (AD)**: ', cpd_AD_vs[0])
            # 3D structure
            st.header('**3D structure of the studied compound:**')
            def makeblock(smi):
                mol = Chem.MolFromSmiles(smi)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                mblock = Chem.MolToMolBlock(mol)
                return mblock

            def render_mol(xyz):
                xyzview = py3Dmol.view()#(width=400,height=400)
                xyzview.addModel(xyz,'mol')
                xyzview.setStyle({'stick':{}})
                xyzview.setBackgroundColor('black')
                xyzview.zoomTo()
                showmol(xyzview,height=500,width=500)
            blk=makeblock(compound_smiles)
            render_mol(blk)
            st.write('You can use the scroll wheel on your mouse to zoom in or out a 3D structure of compound')
                             

    # Read SDF file 
    LOAD = st.sidebar.checkbox('Upload compounds from file sdf')
    if LOAD:
        uploaded_file = st.sidebar.file_uploader("Choose a file")
        if uploaded_file is not None:
            sdfInfo = dict(smilesName='SMILES', molColName='ROMol')
            
            moldf = PandasTools.LoadSDF(uploaded_file, **sdfInfo)
            st.header('**CHECKING STRUCTURES:**')
            st.write('Original data: ', len(moldf), 'molecules')
            # Rename ROMol
            moldf = moldf.rename(columns={'ROMol': 'Mol'})
            # Remove missing RDKit molecules
            moldf = moldf[pd.notnull(moldf['Mol'])]
            if 'StandardizerResult' in moldf.columns:
                moldf = moldf.drop(columns='StandardizerResult')
            # Columns
            st.write('Kept data: ', len(moldf), 'molecules')
            from molvs.validate import Validator
            fmt = '%(asctime)s - %(levelname)s - %(validation)s - %(message)s'
            validator = Validator(log_format=fmt)
            st.write('\n Problematic structures: \n', validator.validate(moldf))

            for i in range(len(moldf.Mol)):
                a= moldf['SMILES']
                b=list(a) 
            vs = []
            for i in range(len(b)):
                m = Chem.MolFromSmiles(b[i])
                vs.append(m) 
           

        
            # Calculate molecular descriptors
            calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
            header = calc.GetDescriptorNames()
            descr_tr= []
            for m in vs:
                descr_tr.append(calc.CalcDescriptors(m))
                X = np.asarray(descr_tr)
                                
            ######################
            # Pre-built model
            ######################

            X_s = scale.transform(X)

            # Apply model to make predictions
            prediction_RF = load_model_RF.predict(X)
            prediction_GBM = load_model_GBM.predict(X)
            prediction_SVM = load_model_SVM.predict(X)
            pred_consensus = 1 * \
            (((prediction_RF + prediction_GBM + prediction_SVM) / 3) >= 0.5)

            pred_consensus = np.array(pred_consensus)
            pred_consensus = np.where(pred_consensus == 1, "Active", "Inactive")
        

            # Estimination AD

            # load numpy array from csv file
            x_tr = loadtxt('RDKit/x_tr.csv', delimiter=',')
            model_AD_limit = 1290022998.45
            neighbors_k_vs = pairwise_distances(x_tr, Y=X, n_jobs=-1)
            neighbors_k_vs.sort(0)
            similarity_vs = neighbors_k_vs
            cpd_value_vs = similarity_vs[0, :]
            cpd_AD_vs = np.where(cpd_value_vs <= model_AD_limit, "Inside AD", "Outside AD")

            st.header('**RESULTS OF PREDICTION:**')
            if st.button('Show results as table'):
                moldf.drop(columns='Mol', inplace=True)
                moldf.drop(columns='ID', inplace=True)                
                pred_beta = pd.DataFrame({'HDAC1 activity': pred_consensus,'Applicability domain (AD)': cpd_AD_vs}, index=None)
                predictions = pd.concat([moldf, pred_beta], axis=1)
                st.dataframe(predictions)           
                def convert_df(df):
                    return df.to_csv().encode('utf-8')  
                csv = convert_df(predictions)

                st.download_button(
                    label="Download results of prediction as CSV",
                    data=csv,
                    file_name='Results.csv',
                    mime='text/csv',
                )

            # Print results
            if st.button('Show  results for each molecule separately'):
                st.header('**Prediction results:**')
                for i in range(len(moldf.Mol)):
                    a= moldf['SMILES']
                    b=list(a)  
                for i in range(len(b)):
                    m = Chem.MolFromSmiles(b[i])
                    im = Draw.MolToImage(m)
                    st.write('**COMPOUNDS NUMBER **' + str(i+1) + '**:**')
                    st.write('**2D structure of compound number **' + str(i+1) + '**:**')
                    st.image(im)
                    # 3D structure
                    st.write('**3D structure of compound number **'+ str(i+1) + '**:**')
                    def makeblock(smi):
                        mol = Chem.MolFromSmiles(smi)
                        mol = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol)
                        mblock = Chem.MolToMolBlock(mol)
                        return mblock

                    def render_mol(xyz):
                        xyzview = py3Dmol.view()#(width=400,height=400)
                        xyzview.addModel(xyz,'mol')
                        xyzview.setStyle({'stick':{}})
                        xyzview.setBackgroundColor('black')
                        xyzview.zoomTo()
                        showmol(xyzview,height=500,width=500)
                    blk=makeblock(b[i])
                    render_mol(blk)
                    st.write('You can use the scroll wheel on your mouse to zoom in or out a 3D structure of compound')

                    st.write('**Smiles for compound number **'+ str(i+1) + '**:**', b[i])
                    st.write('**HDAC1:** ', pred_consensus[i])
                    st.write('**Applicability domain (AD):** ', cpd_AD_vs[i])
                    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    
st.text('© Oleg Tinkov, 2022')
            
