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

# Loading the dataset(s)
#Zheng_data = ad.read_h5ad("/home/data/designproject2122/data/Azimuth PBMC/Zheng_data_labels.h5ad")
Azimuth_data = ad.read_h5ad("/home/data/designproject2122/data/Azimuth PBMC/Azim_data_labels.h5ad")

# downsample the data
Azimuth_data_downsampled = Azimuth_data[Azimuth_data.obs["donor"].isin(['P1', 'P2'])]
Azimuth_data_remaining = Azimuth_data[~Azimuth_data.obs["donor"].isin(['P1', 'P2'])]

from ClassificationModule import ClassificationExperimentOptuna
from optuna import distributions
import optuna

# The pipeline module was already loaded above!
from ScanpyPreprocessingPipeline import PreprocessingHVG, PreprocessingLog1P, PreprocessingNormalize, PreprocessingLINKER, PreprocessingGeneFilter
from ScanpyPreprocessingPipeline import Scaler
# Make a list with the desired preprocessing pipelines
Preprocessing_list = [
    [
        ('linker', PreprocessingLINKER())

    ],

   [
        ('logtransform', PreprocessingLog1P()),
        ('linker', PreprocessingLINKER())
    ]
    ,

    [
        ('logtransform', PreprocessingLog1P()),
        ('normalize_per_cell', PreprocessingNormalize()),
        ('linker', PreprocessingLINKER())
    ]
    ,
    [
        ('logtransform', PreprocessingLog1P()),
        ('normalize_per_cell', PreprocessingNormalize()),
        ('scaler', Scaler()),
        ('linker', PreprocessingLINKER())

    ]
]
#sc.pp.normalize_per_cell(adata)                 # renormalize after filtering
#if log: sc.pp.log1p(adata)                      # log transform: adata.X = log(adata.X + 1)
#sc.pp.scale(adata)                              # scale to unit variance and shift to zero mean

Preprocessing_param_list =  [
    {},
    {},
    {},
    {}
]


# Make a list with the desired classification methods
Classifier_list = [[("decision_tree", DecisionTreeClassifier())],
    [('kneighbors', KNeighborsClassifier())],
    [("rf", RandomForestClassifier(n_jobs=3))],
    [("ridge_log", LogisticRegression(n_jobs=3))],
    [("lasso_log", LogisticRegression(n_jobs=3))],
    [("svc_lin", SVC(kernel='linear'))]
                   
]

Classifier_param_list =  [{"decision_tree__criterion": distributions.CategoricalDistribution(["gini", "entropy"]),
                #"splitter":distributions.CategoricalDistribution(["best","random"]),
 'decision_tree__max_depth':distributions.IntUniformDistribution(2,10),                "decision_tree__min_samples_split":distributions.IntUniformDistribution(2, 100),            "decision_tree__min_samples_leaf":distributions.IntUniformDistribution(1,50)},
                          
{"kneighbors__n_neighbors": distributions.IntUniformDistribution(2, 20)},
                          
{"rf__criterion": distributions.CategoricalDistribution(["gini", "entropy"]),
                "rf__n_estimators":distributions.IntUniformDistribution(10, 300), 
                'rf__max_depth':distributions.IntUniformDistribution(2, 5),
                "rf__min_samples_split":distributions.IntUniformDistribution(2, 100),
                "rf__min_samples_leaf":distributions.IntUniformDistribution(1, 50), 
                "rf__max_samples" :optuna.distributions.UniformDistribution(0.5,1.0),
             "rf__max_features":optuna.distributions.UniformDistribution(0.4,0.85)},
                          
            { "ridge_log__C":distributions.LogUniformDistribution(1e-4,1e4),        "ridge_log__multi_class":distributions.CategoricalDistribution(["multinomial"]),
"ridge_log__penalty": distributions.CategoricalDistribution(["l2"]),
"ridge_log__solver" : distributions.CategoricalDistribution(["newton-cg", "lbfgs"])},
                          
{ "lasso_log__C":distributions.LogUniformDistribution(1e-4,1e4), 
    "lasso_log__multi_class":distributions.CategoricalDistribution(["multinomial"]),
"lasso_log__penalty": distributions.CategoricalDistribution(["l1"]),
"lasso_log__solver" : distributions.CategoricalDistribution(["saga"])},
{"svc_lin__C": distributions.LogUniformDistribution(1e-10, 1e10)}]
n_trails_list = [121, 10, 236, 65, 55, 68,121, 10, 236, 65, 55, 68,121, 10, 236, 65, 55, 68,121, 10, 236, 65, 55, 68]


# This is de name of the column we want to predict. Most likely cell type of course. 
predict_factor = "new_label"

os.chdir('/home/data/designproject2122/notebook/NT/')

group_kfold = GroupKFold(n_splits=2)
iterator = list(group_kfold.split(Azimuth_data_downsampled, Azimuth_data_downsampled.obs["new_label"], Azimuth_data_downsampled.obs["donor"]))
#iterator

experiment = ClassificationExperimentOptuna(Preprocessing_list, Classifier_list,Preprocessing_param_list, Classifier_param_list)

results = experiment.fit(Azimuth_data_downsampled, Azimuth_data_remaining, predict_factor, savemodels=True, cv = iterator, n_trails = n_trails_list, n_jobs=1)
results