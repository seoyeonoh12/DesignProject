from itertools import product, chain
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import numpy as np
import seaborn as sns
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from statistics import mean
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from joblib import dump, load
from datetime import datetime
from copy import deepcopy

from ScanpyPreprocessingPipeline import PreprocessingLog1P, PreprocessingNormalize, PreprocessingLINKER, PreprocessingGeneFilter
import time

class ClassificationExperiment():
    """

    """
    
    #Class Constructor 
    def __init__( self, Preprocessing_list, Classifier_list, cross_validation_splits ):
        # the amount of cross validation splits
        self.cross_validation_splits = cross_validation_splits
        
        # create the pipelines from the provided preprocessings pipelines and classifiers
        self.Pipeline_list = [list(chain.from_iterable(list(product(Preprocessing_list, Classifier_list))[i])) for i in range(len(list(product(Preprocessing_list, Classifier_list))))]
        
        # initiate the empty results dataframe
        self.results_DF = pd.DataFrame(
    columns = ["Fold_Score_" + str(i) for i in range(self.cross_validation_splits)] 
    + ["Mean_Score"] 
    + ["F1_Macro_" + str(i) for i in range(self.cross_validation_splits)]
    + ["Mean_F1_Macro"] 
    + ["F1_Micro_" + str(i) for i in range(self.cross_validation_splits)]
    + ["Mean_F1_Micro"] 
    + ["Fold_Duration_" + str(j) for j in range(self.cross_validation_splits)] 
    + ["Mean_Duration"], index = range(len(self.Pipeline_list)))
        
        # Which preprocessing steps were used
        self.results_DF["Preprocessing_Steps"] = ["+".join([self.Pipeline_list[j][i][0] for i in range(len(self.Pipeline_list[j])-2)]) for j in range(len(self.Pipeline_list))]
        
        # Which classifiers were used 
        self.results_DF["Classifier"] = [self.Pipeline_list[j][-1][0] for j in range(len(self.Pipeline_list))]
        
        # create the datasplit
        self.skf = KFold(n_splits=cross_validation_splits)
        
        # CHANGED FROM stratifiedKFOLD TO KFOLD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Implement using groupkfold when a group is provided!!! TODO!!
        
    
    #Return self nothing else to do here 
    
    # the fitting function
    def fit( self, data, predict_factor, savemodels = False):
        # log time when the function is initated, used for file names
        if savemodels:
            time_prefix = datetime.now().strftime("%Y%m%d_%H:%M:%S")
            with open(str(time_prefix+'_kfold.txt'), 'w') as fp:
                        pass
        
        # predict factor
        self.predict_factor = predict_factor
        
        # loop through the kfold splits
        for count, (train_index, test_index) in enumerate(self.skf.split(data)):
            ## VERANDERD VAN GROUPKFOLD NAAR KFOLD OMDAT ZE NIET ALLEMAAL GROEPEN HEBBEN!!
            print("Fold: " + str(count+1) + " Of: " + str(self.cross_validation_splits))
            
            # creating the train and test dataset for this cycle 
            X_train = data[train_index]
            y_train = X_train.obs[self.predict_factor]
            X_test = data[test_index]
            y_test = X_test.obs[self.predict_factor]
            
            # loop through all pipelines
            for count_methods, classification_method in enumerate(self.Pipeline_list):
                t = time.time()
                
                # actual fitting:
                classification_method = Pipeline(classification_method)
                classification_method.fit(X_train,y_train)
                
                # Save performance scores
                self.results_DF[str("Fold_Score_" + str(count))][count_methods] = classification_method.score(X_test,y_test)
                y_pred = classification_method.predict(X_test)
                self.results_DF[str("F1_Macro_" + str(count))][count_methods] = f1_score(y_test, y_pred, average='macro')
                self.results_DF[str("F1_Micro_" + str(count))][count_methods] = f1_score(y_test, y_pred, average='micro')
                
                ## print some information (THIS COULD BE CLEANED UP)
                print("-------------------")
                print("count_methods: " + str(count_methods) 
                      + ", Fold: " + str(count+1) 
                      + ", Score: "+ str(self.results_DF[str("Fold_Score_" + str(count))][count_methods]))
                print(classification_method)
                elapsed = time.time() - t
                print("Time Elapsed: " + str(elapsed))
                print("-------------------")
                
                # Save fold duration
                self.results_DF[str("Fold_Duration_" + str(count))][count_methods] = elapsed
                
                # safe the models to a file
                if savemodels:
                    temp_file_name = str(time_prefix + "_" 
                                         + self.results_DF["Preprocessing_Steps"][count_methods] + "_" 
                                         + self.results_DF["Classifier"][count_methods] + "_" 
                                         + str(count) + ".joblib")
                    dump(classification_method, temp_file_name)
                    
        # Calculate mean scores for each pipeline            
        self.results_DF["Mean_Score"] = self.results_DF[["Fold_Score_" + str(i) for i in range(self.cross_validation_splits)]].mean(axis = 1)
        self.results_DF["Mean_F1_Macro"] = self.results_DF[["F1_Macro_" + str(i) for i in range(self.cross_validation_splits)]].mean(axis = 1)
        self.results_DF["Mean_F1_Micro"] = self.results_DF[["F1_Micro_" + str(i) for i in range(self.cross_validation_splits)]].mean(axis = 1)
        self.results_DF["Mean_Duration"] = self.results_DF[["Fold_Duration_" + str(i) for i in range(self.cross_validation_splits)]].mean(axis = 1)
        
        return self.results_DF
    
    
    
    
    def Heatmap( self, metric ):
        ax = sns.heatmap(self.results_DF.pivot(index='Classifier', columns='Preprocessing_Steps', values=metric))
        return ax
    
    def SaveResults( self, location ):
        self.results_DF.to_pickle(location)
        
    def LoadResults(self, location ):
        self.results_DF = pd.read_pickle(location)
    
    def GetResults(self):
        return self.results_DF
    

    
    
class ClassificationExperimentCrossValidation():
    """

    """
    
    #Class Constructor 
    def __init__( self, Preprocessing_list, Classifier_list):
        self.Pipeline_list = [list(chain.from_iterable(list(product(Preprocessing_list, Classifier_list))[i])) for i in range(len(list(product(Preprocessing_list, Classifier_list))))]
        self.results_DF = pd.DataFrame(
            columns = ["Score"] + ["Duration"] + ["F1_Macro"] + ["F1_Micro"], 
            index = range(len(self.Pipeline_list))
        )
        self.results_DF["Preprocessing_Steps"] = [" + ".join([self.Pipeline_list[j][i][0] for i in range(len(self.Pipeline_list[j])-2)]) for j in range(len(self.Pipeline_list))]
        self.results_DF["Classifier"] = [self.Pipeline_list[j][-1][0] for j in range(len(self.Pipeline_list))]        
    
    #Return self nothing else to do here    
    def fit( self, fit_data, test_data, predict_factor, savemodels = False):
        # log time when the function is initated, used for file names
        if savemodels:
            time_prefix = datetime.now().strftime("%Y%m%d_%H:%M:%S")
            with open(str(time_prefix+'_CrossDatasetValidation.txt'), 'w') as fp:
                        pass
        
        
        
        self.predict_factor = predict_factor
        X_train = fit_data
        y_train = X_train.obs[self.predict_factor]
        X_test = test_data
        y_test = X_test.obs[self.predict_factor]
        for count_methods, classification_method in enumerate(self.Pipeline_list):
            t = time.time()
            # actual fitting:
            classification_method = Pipeline(classification_method)
            classification_method.fit(X_train,y_train)
            self.results_DF["Score"][count_methods] = classification_method.score(X_test,y_test)
            y_pred = classification_method.predict(X_test)
            self.results_DF["F1_Macro"][count_methods] = f1_score(y_test, y_pred, average='macro')
            self.results_DF["F1_Micro"][count_methods] = f1_score(y_test, y_pred, average='micro')
            
            print(classification_method)
            elapsed = time.time() - t
            print("Time Elapsed: " + str(elapsed))
            print("-------------------")
            self.results_DF["Duration"][count_methods] = elapsed
            
            # safe the models to a file
            if savemodels:
                temp_file_name = str(time_prefix + "_" 
                                     + self.results_DF["Preprocessing_Steps"][count_methods] + "_" 
                                     + self.results_DF["Classifier"][count_methods] + "_" 
                                     + str(count) + ".joblib")
                dump(classification_method, temp_file_name)
        return self.results_DF
    
    
    
    
    def Heatmap( self, metric ):
        ax = sns.heatmap(self.results_DF.pivot(index='Classifier', columns='Preprocessing_Steps', values = metric).astype("float64"))
        return ax
    
    def SaveResults( self, location ):
        self.results_DF.to_pickle(location)
        
    def LoadResults(self, location ):
        self.results_DF = pd.read_pickle(location)
    
    def GetResults(self):
        return self.results_DF

    
    
    
    
from optuna.integration import OptunaSearchCV
from optuna import distributions
from sklearn.metrics import accuracy_score

class ClassificationExperimentOptuna():
    """
    This class is used to perform hyperparameter tuning on a list of classification methods and preprocessing pipelines.
    An experiment object should first be created provided with the list of preprocessing pipelines, classifier pipelines and their parameter ranges.
    experiment = ClassificationExperimentOptuna(Preprocessing_list, Classifier_list,Preprocessing_param_list, Classifier_param_list)

    Actually starting the experiment is done using the .fit function
    fit(fit_data, test_data, predict_factor, savemodels=False, cv = 2, n_trails = 2, cv_group = None)
    -fit_data: the data used for testing
    -test_data: only used for final scores!! NOT USED IN TRAINING!
    -predict_factor: the name of the .obs column that should be predicted
    -savemodels: should the model be saved? location is hardcoded for now, not really implemented
    -cv: this can either be a number: the amound of crossvalidation folds or a CV splitter
    -n_trails: this can either be a number, in this case all models are optimized using the same amound of trails.
        A list can also be provided, here the n_trails is different for every model! (the length of the list should match the total number of trails!! not just the number of methods!!)
    
    """
    
    #Class Constructor 
    def __init__( self, Preprocessing_list, Classifier_list, Preprocessing_param_list, Classifier_param_list):
        self.Preprocessing_param_list = Preprocessing_param_list
        self.Classifier_param_list = Classifier_param_list
        self.total_param_list = [dict(i[0], **i[1]) for i in product(Preprocessing_param_list, Classifier_param_list)]
        self.Pipeline_list = [list(chain.from_iterable(list(product(Preprocessing_list, Classifier_list))[i])) for i in range(len(list(product(Preprocessing_list, Classifier_list))))]
        self.results_DF = pd.DataFrame(
            columns = ["Score"] + ["F1_Macro"] + ["F1_Micro"] + ["Duration"] + ["BestParameters"], 
            index = range(len(self.Pipeline_list))
        )
        self.results_DF["Preprocessing_Steps"] = [" + ".join([self.Pipeline_list[j][i][0] for i in range(len(self.Pipeline_list[j])-2)]) for j in range(len(self.Pipeline_list))]
        self.results_DF["Classifier"] = [self.Pipeline_list[j][-1][0] for j in range(len(self.Pipeline_list))]        
    
    #Return self nothing else to do here    
    def fit( self, fit_data, test_data, predict_factor, savemodels=False, cv = 2, n_trails = 2, n_jobs = 1):
        if savemodels:
            # initialize the time key
            time_prefix = datetime.now().strftime("%Y%m%d_%H:%M:%S")

            workingdir = os.getcwd()
            out_path = os.path.join(workingdir, "Optuna_output")
            # make sure the output folder exists
            try: 
                os.mkdir(out_path) 
            except OSError as error:
                print(error) 
                print("Output Folder Exists?")

            # Make experiment folder
            exp_path = os.path.join(out_path, time_prefix)
            try: 
                os.mkdir(exp_path) 
            except OSError as error:
                print(error) 
                print("YOU SHOULD NEVER SEE THIS MESSAGE!")

            # write empty file with time and name, not really needed anymore?
            with open(os.path.join(exp_path,str(time_prefix+'_Optuna.txt')), 'w') as fp:
                        pass
        
        # split the data
        self.predict_factor = predict_factor
        X_train = fit_data
        y_train = X_train.obs[self.predict_factor]
        X_test = test_data
        y_test = X_test.obs[self.predict_factor]

        for count_methods, classification_method in enumerate(self.Pipeline_list):
            print("STARTING: " + str(count_methods))
            t = time.time()
            if isinstance(n_trails, list):
                n_trails_temp = n_trails[count_methods]
            else:
                n_trails_temp = n_trails

            # actual fitting:
            temp_study = optuna.create_study(direction="maximize")
            classification_method = Pipeline(classification_method)
            param = self.total_param_list[count_methods]
            search = OptunaSearchCV(classification_method, study = temp_study, param_distributions = param, cv = deepcopy(cv), n_trials = n_trails_temp, n_jobs = n_jobs)
            search.fit(X_train, y_train)
            
            # processing results
            self.results_DF["BestParameters"][count_methods] = search.best_params_
            print(search.best_params_)
            y_pred = search.predict(X_test)
            self.results_DF["Score"][count_methods] = accuracy_score(y_test,y_pred)
            self.results_DF["F1_Macro"][count_methods] = f1_score(y_test, y_pred, average='macro')
            self.results_DF["F1_Micro"][count_methods] = f1_score(y_test, y_pred, average='micro')
            print(classification_method)


            # Log time
            elapsed = time.time() - t
            print("Time Elapsed: " + str(elapsed))
            print("-------------------")
            self.results_DF["Duration"][count_methods] = elapsed
            
            
            # Safe results
            if savemodels:
                temp_file_name = str(time_prefix + "_" 
                                     + self.results_DF["Preprocessing_Steps"][count_methods] + "_" 
                                     + self.results_DF["Classifier"][count_methods])

                model_path = os.path.join(exp_path, temp_file_name)
                try: 
                    os.mkdir(model_path) 
                except OSError as error:
                    print(error) 
                    print("YOU SHOULD NEVER SEE THIS MESSAGE!")

                dump(temp_study, os.path.join(model_path,str(temp_file_name+'_study'+".joblib")))
                dump(classification_method, os.path.join(model_path,str(temp_file_name+'_method'+".joblib")))
                dump(search.best_trial_, os.path.join(model_path,str(temp_file_name+'_best_trail'+".joblib")))
                with open(os.path.join(exp_path,str(time_prefix+'_Optuna.txt')), 'a') as fp:
                        fp.write(",".join([
                            temp_file_name, 
                            str(self.results_DF["Score"][count_methods]), 
                            str(self.results_DF["F1_Macro"][count_methods]),
                            str(self.results_DF["F1_Micro"][count_methods]),
                            str(self.results_DF["Duration"][count_methods]),
                            str(self.results_DF["BestParameters"][count_methods])
                            ]))
                        fp.write("\n")
                optuna.visualization.plot_optimization_history(temp_study).write_image(os.path.join(model_path,"History.svg"))
                optuna.visualization.plot_parallel_coordinate(temp_study).write_image(os.path.join(model_path,"Overview.svg"))


        return self.results_DF
    
    
    
    
    def Heatmap( self, metric ):
        ax = sns.heatmap(self.results_DF.pivot(index='Classifier', columns='Preprocessing_Steps', values = metric).astype("float64"))
        return ax
    
    def SaveResults( self, location ):
        self.results_DF.to_pickle(location)
        
    def LoadResults(self, location ):
        self.results_DF = pd.read_pickle(location)
    
    def GetResults(self):
        return self.results_DF



class ClassificationExperimentTrainValidationSplit():
    """

    """
    
    #Class Constructor 
    def __init__( self, Classifier_list):
        self.Classifier_list = Classifier_list
        self.results_DF = pd.DataFrame(
            columns = ["Score"] + ["Duration"] + ["F1_Macro"] + ["F1_Micro"], 
            index = range(len(self.Classifier_list))
        )
  
           
    
    #Return self nothing else to do here    
    def fit( self, fit_data, test_data, predict_factor, savemodels = False):
        # log time when the function is initated, used for file names
        if savemodels:
            time_prefix = datetime.now().strftime("%Y%m%d_%H:%M:%S")
            with open(str(time_prefix+'_CrossDatasetValidation.txt'), 'w') as fp:
                        pass
        
        
        
        self.predict_factor = predict_factor
        X_train = fit_data
        y_train = X_train.obs[self.predict_factor]
        X_test = test_data
        y_test = X_test.obs[self.predict_factor]
        for count_methods, classification_method in enumerate(self.Pipeline_list):
            t = time.time()
            # actual fitting:
            classification_method = Pipeline(classification_method)
            classification_method.fit(X_train,y_train)
            self.results_DF["Score"][count_methods] = classification_method.score(X_test, y_test)
            y_pred = classification_method.predict(X_test)
            self.results_DF["F1_Macro"][count_methods] = f1_score(y_test, y_pred, average='macro')
            self.results_DF["F1_Micro"][count_methods] = f1_score(y_test, y_pred, average='micro')
            
            print(classification_method)
            elapsed = time.time() - t
            print("Time Elapsed: " + str(elapsed))
            print("-------------------")
            self.results_DF["Duration"][count_methods] = elapsed
            
            # safe the models to a file
            if savemodels:
                temp_file_name = str(time_prefix + "_" 
                                     + self.results_DF["Preprocessing_Steps"][count_methods] + "_" 
                                     + self.results_DF["Classifier"][count_methods] + "_" 
                                     + str(count) + ".joblib")
                dump(classification_method, temp_file_name)
        return self.results_DF