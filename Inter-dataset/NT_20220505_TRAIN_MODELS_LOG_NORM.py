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

def FixParameterLibrary(library):
    fixed_library = {}
    for key in library.keys():
        fixed_library[key.split("__")[1]] = library[key]
    return fixed_library

# GET DATA
Zheng_data = ad.read_h5ad("/home/data/designproject2122/data/Azimuth PBMC/Zheng_data_labels.h5ad")
Azimuth_data = ad.read_h5ad("/home/data/designproject2122/data/Azimuth PBMC/Azim_data_labels.h5ad")

Azimuth_training_data = Azimuth_data[Azimuth_data.obs["donor"].isin(['P5', 'P6'])]
Azimuth_test_data = Azimuth_data[~Azimuth_data.obs["donor"].isin(['P5', 'P6'])]

predict_factor = "new_label"
X_train = Azimuth_training_data 
y_train = X_train.obs[predict_factor]
X_test = Azimuth_test_data
y_test = X_test.obs[predict_factor]
X_intra_dataset = Zheng_data
y_intra_dataset = X_intra_dataset.obs[predict_factor]

results_DF = pd.DataFrame(
            columns = ("Name", "Accuracy", "F1_Macro", "F1_Micro", "Duration", "Inta_accuracy", "Intra_F1_Macro", "Intra_F1_Micro")
        )
import os
import joblib
rootdir = '/home/data/designproject2122/notebook/NT/Input/20220501_22:04:31'
outdir = '/home/data/designproject2122/notebook/NT/Output/log_norm'
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
    t = time.time()
    experiment_method.fit(X_train, y_train)

    # TEST MODEL WITHIN DATASET
    y_pred = experiment_method.predict(X_test)
    elapsed = time.time() - t
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    classification_r = classification_report(y_test, y_pred, output_dict=True)


    # TEST MODEL INTRA DATASET
    y_pred_intra = experiment_method.predict(X_intra_dataset)
    accuracy_intra = accuracy_score(y_intra_dataset, y_pred_intra)
    f1_macro_intra = f1_score(y_intra_dataset, y_pred_intra, average='macro')
    f1_micro_intra = f1_score(y_intra_dataset, y_pred_intra, average='micro')
    classification_r_intra = classification_report(y_intra_dataset, y_pred_intra, output_dict=True)

    # OUTPUT DATA
    exp_path = os.path.join(outdir, experiment_name)
    try:
        os.mkdir(exp_path) 
    except OSError as error:
        print(error)

    ## SAVE MODEL
    dump(experiment_method, os.path.join(exp_path, str(experiment_name+"_model.joblib")))

    ## SAVE PREDICTIONS
    dump(y_pred, os.path.join(exp_path, str(experiment_name+"_ypred.joblib")))
    dump(y_pred_intra, os.path.join(exp_path, str(experiment_name+"_ypredintra.joblib")))

    ## SAVE PERFORMANCE
    results_DF.loc[len(results_DF)] = [experiment_name, accuracy, f1_macro, f1_micro, elapsed, accuracy_intra, f1_macro_intra, f1_micro_intra]
    clsf_report = pd.DataFrame(classification_r).transpose()
    clsf_report.to_csv(os.path.join(exp_path, "Class_report.csv"), index= True)

    clsf_report_intra = pd.DataFrame(classification_r_intra).transpose()
    clsf_report_intra.to_csv(os.path.join(exp_path, "Class_report_intra.csv"), index= True)

    print(experiment_name)
    print(experiment_method)
    print(experiment_study)

## SAVE FINAL DATAFRAME
dump(experiment_method, os.path.join(outdir, "OUTDATAFRAME.joblib"))