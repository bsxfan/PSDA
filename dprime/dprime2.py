import numpy as np

def obj(tar_mean,non_mean,var,temp=1/100):
    """
    Minimization-friendly objective. Approaches -log(dprime) when temp>0 is 
    small. 
    
    inputs:
        
        tar_mean: average within-class score
        non_mean: average between-class score
        var: score variance 
             This objective minimizes the given variance, but it is OK if 
             this is pure between-class variance, because for cosine scores,
             the within-class variance is implicitly minimized by the
             maximization of the within-class mean. (Everything is squashed
             against 1). However, mixing in some within-class variance is not
             undesirable. For a large Gram matrix of all pairwise scores in a 
             large dataset, the between-class scores will dominate and this is 
             OK. 
                                        
            
            
        temp >= 0
    
        - Any temp > 0 prevents the objective from blowing up
          when dprime is near zero.
        
        - Any temp > 2 prevents the objective from returning NaN when
          dprime is negative.
          
        - 0 < temp << 1 approaches -log(dprime)
        
    returns to-be-minimized objective    

    """
    num = np.log(var) 
    den = np.log1p((tar_mean-non_mean)/temp) + np.log(temp)
    return num-den
