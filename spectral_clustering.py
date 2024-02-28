from scipy.sparse.linalg import eigsh
import scipy
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
from scipy.sparse import linalg

device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device.type=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    


class Spectral_clustering_init():
    def __init__(self,num_of_eig=7):
        
        self.num_of_eig=num_of_eig

    
    
    
    def spectral_clustering(self,mask_i=None,mask_j=None,UN=False):
        
        if UN:
            sparse_i=mask_i[self.sparse_i_idx].cpu().numpy()
            sparse_j=mask_j[self.sparse_j_idx].cpu().numpy()
          
            sparse_i_new=np.concatenate((sparse_i,sparse_j))
            sparse_j_new=np.concatenate((sparse_j,sparse_i))
                
            sparse_i=sparse_i_new
            sparse_j=sparse_j_new
                
            V=np.ones(sparse_i.shape[0])
       
            Affinity_matrix=sparse.coo_matrix((V,(sparse_i,sparse_j)),shape=(self.NN,self.NN))
            eig_val, eig_vect = scipy.sparse.linalg.eigsh(Affinity_matrix,self.num_of_eig,which='LM')
            X = eig_vect.real
            rows_norm = np.linalg.norm(X, axis=1, ord=2)
            U_norm = (X.T / rows_norm).T
            return torch.from_numpy(U_norm).float().to(device)
            
        else:
            sparse_i=self.sparse_i_idx.cpu().numpy()
            sparse_j=self.sparse_j_idx.cpu().numpy()
            
                
            V=np.ones(sparse_i.shape[0])
       
            self.Affinity_matrix=sparse.coo_matrix((V,(sparse_i,sparse_j)),shape=(self.input_size_1,self.input_size_2))
            u,_,v=linalg.svds(self.Affinity_matrix, k=self.num_of_eig)
            u=np.array(u)
            v=np.array(v)
    
            return torch.from_numpy(u).float().to(device),torch.from_numpy(v.transpose()).float().to(device)
        
