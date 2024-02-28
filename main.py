# Import all the packages
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


import pandas as pd

CUDA = torch.cuda.is_available()
from spectral_clustering import Spectral_clustering_init
from sklearn import metrics
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device.type=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    
    
undirected=1
import matplotlib.pyplot as plt

class LSM(nn.Module,Spectral_clustering_init):
    def __init__(self,means_,stds_,T_i,T_j,data,sparse_i,sparse_j, input_size_1,input_size_2,latent_dim):
        super(LSM, self).__init__()
        Spectral_clustering_init.__init__(self,num_of_eig=latent_dim)
        self.input_size_1=input_size_1
        self.input_size_2=input_size_2
        self.input_size=input_size_1
        self.latent_dim=latent_dim
       
        # Initialize latent space with the centroids provided from the Fractal over the spectral clustering space
        #self.kmeans_tree_recursively(depth=80,init_first_layer_cent=self.first_centers)
        self.bias=nn.Parameter(torch.randn(1,device=device))
        self.scaling_factor=nn.Parameter(torch.randn(1,device=device))
        self.T_i=T_i
        self.T_j=T_j        
        
        self.initialization=1
        self.scaling=1
        #create indices to index properly the receiver and senders variable
        self.flag1=0
        self.pdist = nn.PairwiseDistance(p=2,eps=0)
        self.softplus=nn.Softplus()
        
        self.sparse_i_idx=sparse_i.long()

        self.sparse_j_idx=sparse_j.long()


        self.sampling_i_weights=torch.ones(input_size_1)
        self.sampling_j_weights=torch.ones(input_size_2)

        
      
             
          
        self.weights_=torch.ones(sparse_i.shape[0])
            
      
           
        
        
        spectral_z,spectral_w=self.spectral_clustering()
       
        self.latent_z=nn.Parameter(spectral_z)
        self.latent_w=nn.Parameter(spectral_w)
        
        # Random effects citing and cited
        self.gamma=nn.Parameter(torch.randn(self.input_size_1,device=device))
        
        self.alpha=nn.Parameter(torch.randn(self.input_size_2,device=device))
        
        # trunc normal
        self.alpha_trunc=torch.zeros(self.input_size_2,device=device)#

        self.pi=torch.tensor(np.pi,device=device)
        if dist=="Truncated":
            self.mean_pre=nn.Parameter(torch.log(torch.exp(means_)-1))
            self.sigma_pre=nn.Parameter(torch.log(torch.exp(stds_)-1))
        elif dist=="Log-normal":
            self.mean_pre=nn.Parameter(means_)
            self.sigma_pre=nn.Parameter(stds_)


        self.sigma_pre=nn.Parameter(torch.rand(self.input_size_2,device=device))
        
    # def sample_network(self):
    #     # USE torch_sparse lib i.e. : from torch_sparse import spspmm

    #     # sample for bipartite network
    #     sample_i_idx = torch.multinomial(self.sampling_i_weights, self.sample_i_size, replacement=False)
    #     sample_j_idx = torch.multinomial(self.sampling_j_weights, self.sample_j_size, replacement=False)
    #     # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
    #     indices_i_translator = torch.cat([sample_i_idx.unsqueeze(0), sample_i_idx.unsqueeze(0)], 0)
    #     indices_j_translator = torch.cat([sample_j_idx.unsqueeze(0), sample_j_idx.unsqueeze(0)], 0)
    #     # adjacency matrix in edges format
    #     edges = torch.cat([self.sparse_i_idx.unsqueeze(0), self.sparse_j_idx.unsqueeze(0)], 0)
    #     # matrix multiplication B = Adjacency x Indices translator
    #     # see spspmm function, it give a multiplication between two matrices
    #     # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
    #     indexC, valueC = spspmm(edges, torch.ones(self.sparse_i_idx.shape[0]), indices_j_translator,
    #                             torch.ones(indices_j_translator.shape[1]), self.input_size_1, self.input_size_2,
    #                             self.input_size_2, coalesced=True)
    #     # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
    #     indexC, valueC = spspmm(indices_i_translator, torch.ones(indices_i_translator.shape[1]), indexC, valueC,
    #                             self.input_size_1, self.input_size_1, self.input_size_2, coalesced=True)

    #     # edge row position
    #     sparse_i_sample = indexC[0, :]
    #     # edge column position
    #     sparse_j_sample = indexC[1, :]
       
    #     return sample_i_idx, sample_j_idx, sparse_i_sample, sparse_j_sample, valueC
    
    def init_case_control(self,control_size=5):
        
        # in this formulation i cite j, i->j, meaning that T_i>T_j
        # both T_i and T_j are sorted in time
        # will contain after which data length a paper can be cited and thus have a likelihood contribution
        list_i=[]

        index=0
        for t__ in self.T_j:
            for j in range(index,self.T_i.shape[0]):
                # due to sorting in time we can continue from the next indexing and avoid unessecary comparisons
                # index holds where the previous paper stopped at and where the next one starts
                if t__<=self.T_i[j]:
                    list_i.append(index)
                    break
                else:
                    index+=1
                
        self.sampling_from=torch.tensor(list_i)
                
        self.non_link_j=torch.repeat_interleave(torch.arange(self.input_size_2),control_size*self.degree_j)

        self.non_link_i=torch.repeat_interleave(self.sampling_from,control_size*self.degree_j)

     

    def sampling_case_control(self):

        pair_i=self.non_link_i+torch.ceil(torch.rand(self.non_link_i.shape[0])*(self.input_size_1-1-self.non_link_i))
        
        s1 = torch.sparse_coo_tensor(torch.cat((self.sparse_i_idx.unsqueeze(0),self.sparse_j_idx.unsqueeze(0)),0), torch.ones(self.sparse_i_idx.shape[0]), (self.input_size_1, self.input_size_2))

        s2 = torch.sparse_coo_tensor(torch.cat((pair_i.unsqueeze(0),self.non_link_j.unsqueeze(0)),0), torch.ones(pair_i.shape[0]), (self.input_size_1, self.input_size_2)).coalesce()


        extra=s1*s2
        
        s_final=(s2-extra).coalesce()
        self.s_final=s_final
        
        final_i,final_j=s_final._indices()
        final_times=s_final._values()
        control_degrees=torch.sparse.sum(s_final,0)._values()
        
        control_weights=self.input_size_1-self.sampling_from-self.degree_j
        
        control_weights=control_weights/control_degrees
        
        
        control_weights=control_weights[final_j]


        
        return final_i,final_j,final_times,control_weights
      


   
    def phi_x(self,x_t):
        phi=(1/torch.sqrt(2*self.pi))*torch.exp(-0.5*(x_t*x_t))
        return phi
    
    def PHI_x(self,x_t):
        PHI=0.5*(1+torch.special.erf((x_t)/(2**0.5)))
        return PHI
    
    def CDF_trun(self):
        max_T=self.T_i.max()
        T_j=(max_T+1e-06)-self.T_j
        x_mu_sigma=(T_j-self.mean)/self.sigma
        nom=self.PHI_x(x_mu_sigma)-self.PHI_x(x_t=((self.alpha_trunc-self.mean)/self.sigma))
        #print(nom.min())
        Z=1-self.PHI_x(x_t=((self.alpha_trunc-self.mean)/self.sigma))
        CDF=nom/Z
        return CDF.float(),T_j
    def PDF_trunc(self,sparse_j_sample,sparse_i_sample):
        x_t_link=torch.abs(self.T_i[sparse_i_sample]-self.T_j[sparse_j_sample])
        phi_x_t=(x_t_link-self.mean[sparse_j_sample])/(self.sigma[sparse_j_sample]+1e-06)
        PDF=1e-06+((1/(self.sigma[sparse_j_sample]+1e-06))*(self.phi_x(x_t=phi_x_t)/(1-self.PHI_x(x_t=((self.alpha_trunc[sparse_j_sample]-self.mean[sparse_j_sample])/(self.sigma[sparse_j_sample]+1e-06))))))
        return PDF
    
    def CDF_log(self):
        max_T=self.T_i.max()
        T_j=(max_T+1e-06)-self.T_j
        #print(nom.min())
        CDF=self.PHI_x(x_t=((torch.log(T_j)-self.mean)/(self.sigma+1e-06)))
        return CDF.float(),T_j
    
    def PDF_log(self,sparse_j_sample,sparse_i_sample):
        x_t_link=torch.abs(self.T_i[sparse_i_sample]-self.T_j[sparse_j_sample])+1e-06

        log_pdf=-torch.log(x_t_link)-torch.log(self.sigma[sparse_j_sample]+1e-06) - 0.5*torch.log(2*self.pi)-(1/(2*self.sigma[sparse_j_sample]**2+1e-06))*((torch.log(x_t_link)-self.mean[sparse_j_sample])**2)
        return log_pdf
    
    def CDF_gen(self,test_time,test_item,dist= 'Truncated'):
        if dist== 'Truncated':

            max_T=self.T_i.max()
            T_j=(max_T+1e-06)-test_time
            x_mu_sigma=(T_j[test_item]-self.mean[test_item])/self.sigma[test_item]
            nom=self.PHI_x(x_mu_sigma[test_item])-self.PHI_x(x_t=((self.alpha_trunc[test_item]-self.mean[test_item])/self.sigma[test_item]))
            #print(nom.min())
            Z=1-self.PHI_x(x_t=((self.alpha_trunc[test_item]-self.mean[test_item])/self.sigma[test_item]))
            CDF=nom/Z
        else:
            max_T=self.T_i.max()
            T_j=(max_T+1e-06)-test_time
            #print(nom.min())
            CDF=self.PHI_x(x_t=((torch.log(T_j[test_item])-self.mean[test_item])/(self.sigma[test_item]+1e-06)))
            
        return CDF
    
    
    
    
    
    
    def pdf_optim(self,epoch,dist='Truncated'):
       
        self.epoch=epoch
        sample_i_idx,sample_j_idx,final_times,control_weights=self.sampling_case_control()
        
        if dist=='Truncated':
            self.sigma=self.softplus(self.sigma_pre)
           
            self.mean=self.softplus(self.mean_pre)
            
            CDF,T_j=self.CDF_trun()
            PDF=self.PDF_trunc(self.sparse_j_idx,self.sparse_i_idx)
            
        elif dist=='Log-normal':
            self.sigma=self.softplus(self.sigma_pre)
            
            self.mean=self.mean_pre
            CDF,T_j=self.CDF_log()
            log_PDF=self.PDF_log(self.sparse_j_idx,self.sparse_i_idx)


        

        if dist=='Truncated':

            logit_u=torch.log(PDF)
        elif dist=='Log-normal':
            logit_u=log_PDF

       
    
   
        
        #########################################################################################################################################################      
        log_likelihood_sparse=torch.sum(logit_u)
        #############################################################################################################################################################        
                 
            
        return log_likelihood_sparse

    #introduce the likelihood function containing the two extra biases gamma_i and alpha_j
    def LSM_likelihood_bias(self,epoch,dist='Truncated'):
        
        self.epoch=epoch
        sample_i_idx,sample_j_idx,final_times,control_weights=self.sampling_case_control()
        
        if dist=='Truncated':
            self.sigma=self.softplus(self.sigma_pre)
            self.mean=self.softplus(self.mean_pre)
            
            CDF,T_j=self.CDF_trun()
            PDF=self.PDF_trunc(self.sparse_j_idx,self.sparse_i_idx)
            
        elif dist=='Log-normal':
            self.sigma=self.softplus(self.sigma_pre)
            self.mean=self.mean_pre
            CDF,T_j=self.CDF_log()
            log_PDF=self.PDF_log(self.sparse_j_idx,self.sparse_i_idx)


        

        if self.scaling:

            
            if dist=='Truncated':
    
                logit_u=self.gamma[self.sparse_i_idx]+self.alpha[self.sparse_j_idx]+torch.log(PDF).detach()
            elif dist=='Log-normal':
                logit_u=self.gamma[self.sparse_i_idx]+self.alpha[self.sparse_j_idx]+log_PDF.detach()

            non_link_link=torch.log(1+torch.exp(self.gamma[self.sparse_i_idx]+self.alpha[self.sparse_j_idx])*((CDF.detach())[self.sparse_j_idx]))
           
            non_link_u=control_weights*final_times*torch.log(1+torch.exp(self.gamma[sample_i_idx]+self.alpha[sample_j_idx])*((CDF.detach())[sample_j_idx]))
        else:
            


            z_pdist=(((self.latent_z[self.sparse_i_idx]-self.latent_w[self.sparse_j_idx]+1e-06)**2).sum(-1))**0.5
            if dist=='Truncated':
                logit_u=-z_pdist+self.gamma[self.sparse_i_idx]+self.alpha[self.sparse_j_idx]+torch.log(PDF)
            elif dist=='Log-normal':
                logit_u=-z_pdist+self.gamma[self.sparse_i_idx]+self.alpha[self.sparse_j_idx]+log_PDF

     
            
            mat=(((self.latent_z[sample_i_idx]-self.latent_w[sample_j_idx]+1e-06)**2).sum(-1))**0.5
            
            non_link_link=torch.log(1+torch.exp(self.gamma[self.sparse_i_idx]+self.alpha[self.sparse_j_idx]-z_pdist)*CDF[self.sparse_j_idx])

            non_link_u=control_weights*final_times*torch.log(1+torch.exp(self.gamma[sample_i_idx]+self.alpha[sample_j_idx]-mat)*CDF[sample_j_idx])
   
        
            
            
        
                
        ####################################################################################################################################
                
                                
                #take the sampled matrix indices in order to index gamma_i and alpha_j correctly and in agreement with the previous
                #remember the indexing of the z_pdist vector
               
        
        #########################################################################################################################################################      
        log_likelihood_sparse=torch.sum(logit_u)-non_link_u.sum()-non_link_link.sum()
        #############################################################################################################################################################        
                 
            
        return log_likelihood_sparse
    
    
    
   
    def gen_gamma_generate(self,x,item,dist='Truncated'):
      
        item_id=item       
        if dist=='Truncated':
    
            x_t=x
            phi_x_t=(x_t-self.mean[item_id])/(self.sigma[item_id]+1e-06)
            PDF=(1/self.sigma[item_id])*(self.phi_x(x_t=phi_x_t)/(1-self.PHI_x(x_t=((self.alpha_trunc[item_id]-self.mean[item_id])/(self.sigma[item_id]+1e-06)))))
        elif dist=='Log-normal':
            PDF=(1/((x+1e-06)*self.sigma[item_id]*torch.sqrt(2*self.pi)+1e-06))*torch.exp(-(((torch.log(x+1e-06)-self.mean[item_id])**2)/(2*(self.sigma[item_id]**2)+1e-06)))
    
        return PDF
    
    def link_prediction_degree(self,target,test_i,test_j):

        
        with torch.no_grad():
           
    
    
            rates=self.degree_i[test_i]*self.degree_j[test_j]
            
            precision, tpr, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

   
        return metrics.roc_auc_score(target.cpu().data.numpy(),rates.cpu().data.numpy()),metrics.auc(tpr,precision)
    
    def link_prediction(self,target,test_i,test_j,pdf):

        
        with torch.no_grad():
           
    
            z_pdist=(((self.latent_z[test_i]-self.latent_w[test_j]+1e-06)**2).sum(-1))**0.5
    
            rates=torch.exp(self.gamma[test_i]+self.alpha[test_j]-z_pdist)*pdf
            
            precision, tpr, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

   
        return metrics.roc_auc_score(target.cpu().data.numpy(),rates.cpu().data.numpy()),metrics.auc(tpr,precision)
   
   



   
    

    

latent_dims=[1,2,3,8]
dists=["Truncated"]

datasets=["SOC"]#'reuters','github',

for dist in dists:
    for dataset in datasets:
       
        for cv_split in range(1):
         
            
            
            
           
                
            for latent_dim in latent_dims:
                print(dataset)
                losses=[]
                ROC=[]
                PR=[]
                zetas=[]
                betas=[]
                scalings=[]
               
        
        
        # ################################################################################################################################################################
        # ################################################################################################################################################################
        # ################################################################################################################################################################
               
                
                
                
                sparse_i=pd.read_csv('./datasets/'+dataset+'/samples/sparse_i.txt',header=None)
                sparse_i=torch.tensor(sparse_i.values.reshape(-1),device=device).long()
                N1=int(sparse_i.max()+1)
                sparse_j=pd.read_csv('./datasets/'+dataset+'/samples/sparse_j.txt',header=None)-N1
                sparse_j=torch.tensor(sparse_j.values.reshape(-1),device=device).long()
                            #del edges
                N2=int(sparse_j.max()+1)
        
                time_i=np.loadtxt('./datasets/'+dataset+'/samples/date_i.txt')
                time_j=np.loadtxt('./datasets/'+dataset+'/samples/date_j.txt')
                min_time=min(time_i.min(),time_j.min())
                time_i=time_i-min_time
                time_j=time_j-min_time
    
                
                max_time=max(time_i.max(),time_j.max())
                time_of_node_i=torch.tensor(time_i.reshape(-1),device=device)
                time_of_node_j=torch.tensor(time_j.reshape(-1),device=device)
    
                time_of_node_i=10*(time_of_node_i/max_time)+1e-06
                time_of_node_j=10*(time_of_node_j/max_time)
                
                
                
                pos_i=pd.read_csv('./datasets/'+dataset+'/samples/pos_i.txt',header=None)
                pos_i=torch.tensor(pos_i.values.reshape(-1),device=device).long()
                pos_j=pd.read_csv('./datasets/'+dataset+'/samples/pos_j.txt',header=None)-N1 
                pos_j=torch.tensor(pos_j.values.reshape(-1),device=device).long()
                
                neg_i=pd.read_csv('./datasets/'+dataset+'/samples/neg_i.txt',header=None)
                neg_i=torch.tensor(neg_i.values.reshape(-1),device=device).long()
                neg_j=pd.read_csv('./datasets/'+dataset+'/samples/neg_j.txt',header=None)-N1 
                neg_j=torch.tensor(neg_j.values.reshape(-1),device=device).long()
                
                target=torch.cat((torch.ones(pos_i.shape[0]), torch.zeros(neg_i.shape[0])))
                test_i=torch.cat((pos_i, neg_i))
                test_j=torch.cat((pos_j, neg_j))
                test_time=time_of_node_i[test_i]-time_of_node_j[test_j]
                
                means=[]
                stds=[]
                for j in sparse_j.unique():
                    T_j=time_of_node_j[j]
                    T_i=time_of_node_i[sparse_i[torch.where(sparse_j==j)[0]]]
                    d_T_j=T_i-T_j
                    means.append(d_T_j.mean())
                    stds.append(d_T_j.std())
                means_=torch.stack(means)
        
                stds_=torch.stack(stds)
                if dist=="Log-normal":
                    # Compute parameters mu and sigma for the normal distribution of ln(X)
                    mu = torch.log(means_)-0.5*torch.log((stds_/means_)**2+1)
                    sigma = torch.sqrt(torch.log(1 + (stds_ / means_)**2))

                    # Compute the mean and std deviation of the log-normal distribution
                    means_ = mu
                    stds_ = sigma
               
    
                
               
                model = LSM(means_,stds_,time_of_node_i,time_of_node_j,torch.randn(N1,latent_dim),sparse_i,sparse_j,N1,N2,latent_dim=latent_dim).to(device)
         
                
                model.degree_j=sparse_j.unique(return_counts=True)[1]
                
                model.degree_i=sparse_i.unique(return_counts=True)[1]

                print('SIZE: ',N1)
                # print('sample size_i: ',model.sample_i_size)
                print('SIZE: ',N2)
                # print('sample size_i: ',model.sample_j_size)
    
                #model = LSM(torch.randn(N,latent_dim),sparse_i,sparse_j,N,latent_dim=latent_dim,CVflag=False,graph_type='undirected',missing_data=False).to(device)
                lr=0.1
                optimizer = optim.Adam(model.parameters(), lr) 
                
                model.init_case_control()
    
                
                rocs=[]
                prs=[]
                dets=[]
    
                #model.load_state_dict(torch.load(f'./model_PDF_{dist}'))
    
                for epoch in range(3001):
                    if epoch==1000:
                        model.scaling=0
                    if epoch<500:
                        loss=-model.pdf_optim(epoch=epoch,dist=dist)#model.LSM_likelihood_bias(epoch=epoch,dist=dist)
                    else:
                        loss=-model.LSM_likelihood_bias(epoch=epoch,dist=dist)#model.LSM_likelihood_bias(epoch=epoch,dist=dist)
    
    
         
            
             
                    optimizer.zero_grad() # clear the gradients.   
                    loss.backward() # backpropagate
                    optimizer.step() # updat
                    #print("degree", model.link_prediction_degree(target, test_i, test_j))
                    if epoch%500==0:
                        print(epoch)
                    if epoch%500==0:
                        # model.knn_time()
                        
                        pdf=model.gen_gamma_generate(x=test_time, item=test_j,dist=dist)
                        cdf=model.CDF_gen(test_time,test_j,dist= dist)
    
                        if epoch>=500:
                            roc,pr=model.link_prediction(target,test_i,test_j,pdf)
                            
                            
                            print(f"AUC-PR SCORE Perfomance in {epoch} model iterations: {pr}")
                            print(f"AUC-ROC SCORE Perfomance in {epoch} model iterations: {roc}")

                            
                    
                        x=torch.linspace(0,10,40)
                        if dist=='Truncated':
        
                            model.sigma=model.softplus(model.sigma_pre)
                           
                            model.mean=model.softplus(model.mean_pre)
                            CDF,T_j=model.CDF_trun()
    
                        elif dist=='Log-normal':
                            model.sigma=model.softplus(model.sigma_pre)
                           
                            model.mean=model.mean_pre
                            CDF,T_j=model.CDF_log()
    
                        print('#############################################################################################')
                        print('#############################################################################################')
                        for i in range(10):
                            item=(torch.ones(1)*np.random.randint(0,N2,(1,))[0]).long()
                            
                            idxt=torch.where(sparse_j==item)
                            # who cites it
                            i_pos=sparse_i[idxt[0]]
                            # timestamp of paper
                            t_item_j=time_of_node_j[item]
                            t_item_i=time_of_node_i[i_pos]
                            d_t=(t_item_i-t_item_j).abs()
                            
                            print('degree: ', d_t.shape[0])
                            
                            times_t=np.ones(d_t.shape)
                            if d_t.shape[0]>0:
                                fig, ax = plt.subplots()
    
                                ax.hist(d_t.cpu().numpy(),range=(d_t.cpu().numpy().min(),10),bins=40,density=True) 
                                #plt.show()         
                                gen_gamma=model.gen_gamma_generate(x=x, item=int(item),dist=dist)
                                ax.plot((x).cpu().numpy(),gen_gamma.detach().cpu().numpy())
                            
                                plt.show()
                                plt.close('all')
    
                torch.save(model.state_dict(), f'./models/Impact_model_PDF_{dist}_{dataset}_{cv_split}_{latent_dim}')
            
            
            


    
            
            
