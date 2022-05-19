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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

def FixParameterLibrary(library):
    fixed_library = {}
    for key in library.keys():
        fixed_library[key.split("__")[1]] = library[key]
    return fixed_library

# GET DATA
Zheng_data = ad.read_h5ad("/home/data/designproject2122/data/Azimuth PBMC/Zheng_data_labels.h5ad")



results_DF = pd.DataFrame(
            columns = ("Name", "Performance")
        )
import os
import joblib
rootdir = '/home/data/designproject2122/notebook/NT/Input/All_models'
outdir = '/home/data/designproject2122/notebook/NT/Output/ZHENG'
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

    # CREATE MODEL
    experiment_method[-1].set_params(**FixParameterLibrary(experiment_study.best_trial.params))

    # TRAIN MODEL
    gss = KFold(n_splits=4, random_state=99, shuffle=True)
    cv_gen = gss.split(Zheng_data, Zheng_data.obs['new_label'])    
    scoring = ['accuracy', 'f1_micro', 'f1_macro']
    scores = cross_validate(experiment_method, Zheng_data, Zheng_data.obs['new_label'], scoring=scoring, cv = cv_gen)

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
dump(results_DF, os.path.join(outdir, "OUTDATAFRAME_ZHENG.joblib"))