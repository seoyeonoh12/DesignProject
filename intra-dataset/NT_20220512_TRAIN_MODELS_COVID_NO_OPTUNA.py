# Loading a bunch of packages! TODO: This should be cleaned up
import pandas as pd
import numpy as np
import scanpy as sc
import scprep
import time
import os
import sys

from ScanpyPreprocessingPipeline import PreprocessingLog1P, PreprocessingNormalize, PreprocessingLINKER, PreprocessingGeneFilter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
import anndata as ad

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from statistics import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_is_fitted

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import cross_validate

def FixParameterLibrary(library):
    fixed_library = {}
    for key in library.keys():
        fixed_library[key.split("__")[1]] = library[key]
    return fixed_library

# GET DATA
# DATASET SPECIFIC STUFF (expected: 12 seconds)
# IN THIS STEP THE DATASET SHOULD BE PUT IN A STANDARD FORMAT TO INPUT INTO THE PIPELINE
t = time.time()
predict_factor = "Factor Value[inferred cell type - ontology labels]"
path = "/home/data/designproject2122/data/covid19_RNAseq"
os.chdir(path)

annot = pd.read_csv('ExpDesign-E-MTAB-9221.tsv', sep='\t')
data = sc.read("E-MTAB-9221.aggregated_filtered_counts.mtx")
cols = pd.read_csv("E-MTAB-9221.aggregated_filtered_counts.mtx_cols", sep="\t", header=None)
rows = pd.read_csv("E-MTAB-9221.aggregated_filtered_counts.mtx_rows", sep="\t", header=None)
data = data.T

rows.drop(rows.columns[1], axis = 1, inplace = True)
          
df_celltypes = annot[['Assay','Factor Value[inferred cell type - ontology labels]','Sample Characteristic[sex]','Sample Characteristic[age]', 'Sample Characteristic[individual]','Factor Value[sampling time point]','Factor Value[clinical history]','Factor Value[disease]']]
df_celltypes = df_celltypes.set_index('Assay')
data.obs = df_celltypes
          
from pyensembl import EnsemblRelease
# run in console: pyensembl install --release 77 --species homo_sapiens
gene_data = EnsemblRelease(77)

temp_list = list()
for count, temp_gene in enumerate(rows[0].tolist()):
    try:
        gene_data.gene_name_of_gene_id(temp_gene)
    except:
        temp_list.append(float("NaN"))
    else:
        temp_list.append(gene_data.gene_name_of_gene_id(temp_gene))

    

rows[1] = temp_list
rows[1].isna().sum() ## 311 genes were not found??? The method I used was very sketchy 

rows.columns = ["GeneID", "GeneName"]
rows = rows.set_index('GeneID')

data.var = rows
data
data_all = data.copy()

# Filter out data that does not contain the a label for the factor of interest:
data = data_all[data_all.obs[predict_factor].notna()]

# label MT genes
data.var['mt'] = data.var["GeneName"].str.startswith('MT-')
data.var.loc[data.var['mt'].isna(), "mt"] = False


results_DF = pd.DataFrame(
            columns = ("Name", "Performance")
        )
import os
import joblib
rootdir = '/home/data/designproject2122/notebook/NT/Input/All_models'
outdir = '/home/data/designproject2122/notebook/NT/Output/COVID_NO_OPTUNA'
subfolders = [ f.path for f in os.scandir(rootdir) if f.is_dir() ]
subfolders
for experimentfolder in subfolders:
    experiment_name = experimentfolder.split("/")[-1]

    # GET MODEL AND SETTINGS
    for files in os.scandir(experimentfolder):
        if files.path.endswith("method.joblib"):
            with open(files.path, 'rb') as fo:
                experiment_method = joblib.load(fo)
        elif files.path.endswith("study.joblib"):
            with open(files.path, 'rb') as fo:
                experiment_study = joblib.load(fo)



    # TRAIN MODEL
    gss = GroupShuffleSplit(n_splits=4, test_size=0.2, random_state=99)
    cv_gen = gss.split(data, data.obs['Factor Value[inferred cell type - ontology labels]'], groups=data.obs['Sample Characteristic[individual]'])    
    scoring = ['accuracy', 'f1_micro', 'f1_macro']
    scores = cross_validate(experiment_method, data, data.obs['Factor Value[inferred cell type - ontology labels]'], scoring=scoring, cv = cv_gen, n_jobs = 1)

    # OUTPUT DATA
    exp_path = os.path.join(outdir, experiment_name)
    try:
        os.mkdir(exp_path) 
    except OSError as error:
        print(error)

    ## SAVE MODEL
    dump(experiment_method, os.path.join(exp_path, str(experiment_name+"_model.joblib")))

    ## SAVE PERFORMANCE
    results_DF.loc[len(results_DF)] = [experiment_name, scores]

    print(experiment_name)
    print(experiment_method)
    print(experiment_study)
print(results_DF)
## SAVE FINAL DATAFRAME
dump(results_DF, os.path.join(outdir, "OUTDATAFRAME_COVID.joblib"))