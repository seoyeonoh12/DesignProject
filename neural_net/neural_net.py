import optuna
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from sklearn.metrics import accuracy_score
class Model(nn.Module):
    def __init__(self, nlayers,dropout,hidden_size1=None, hidden_size2=None, 
                 hidden_size3 = None, hidden_size4 = None, nfeatures=NUM_FEATURES, ntargets=NUM_CLASSES):
        super().__init__()
        layers = []
        hidden_size = [hidden_size1, hidden_size2, hidden_size3, hidden_size4] # loops are great
        for i in range(nlayers): 
            if len(layers) == 0: # initialise
                layers.append(nn.Linear(nfeatures, hidden_size[i]))
                layers.append(nn.BatchNorm1d(hidden_size[i]))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
            else: # keep adding layer over size nlayers
                layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
                layers.append(nn.BatchNorm1d(hidden_size[i]))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size[i],ntargets))
        self.model = nn.Sequential(*layers)
    def forward(self,x):
        return self.model(x)