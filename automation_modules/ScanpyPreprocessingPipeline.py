from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import scanpy as sc
import scprep
from sklearn.preprocessing import StandardScaler, RobustScaler


class PreprocessingLog1P( BaseEstimator, TransformerMixin ):
    """

    """
    
    #Class Constructor 
    def __init__( self ):
        pass
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        X1 = sc.pp.log1p(X, copy = True)
        return X1
    

class PreprocessingNormalize( BaseEstimator, TransformerMixin ):
    """

    """
    
    #Class Constructor 
    def __init__( self ):
        pass
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        sc.pp.normalize_total(X)
        return X



class Scaler( BaseEstimator, TransformerMixin ):
    """
    """
        
    #Class Constructor 
    def __init__( self ):
        pass
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        sc.pp.scale(X, zero_center=False)
        return X




class PreprocessingLINKER( BaseEstimator, TransformerMixin ):
    """

    """
    
    #Class Constructor 
    def __init__( self ):
        pass
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X.X
    
    
class PreprocessingGeneFilter( BaseEstimator, TransformerMixin ):
    """
        
    """
    
    #Class Constructor 
    def __init__( self, min_cells ):
        self.min_cells = min_cells
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        ## filter genes
        self.keep = sc.pp.filter_genes(X, min_cells=self.min_cells, inplace = False)[0]
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        check_is_fitted(self, ['keep'])
        return X[:,self.keep]
    
class PreprocessingHVG( BaseEstimator, TransformerMixin ):
    """
        HASN'T BEEN CHECKED YET!!! STILL NEEDS TO BE DONE
        
    """
    
    #Class Constructor 
    def __init__( self, n_top_genes, flavor ):
        # default values: 0.0125, 3, 0.5
        self.n_top_genes = n_top_genes
        self.flavor = flavor
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        ## filter genes
        self.keep = sc.pp.highly_variable_genes(X, n_top_genes=self.n_top_genes, flavor=self.flavor, inplace = False)["highly_variable"]
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        check_is_fitted(self, ['keep'])
        return X[:,self.keep]