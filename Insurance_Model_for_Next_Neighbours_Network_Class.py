#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from NeighbourClustering import *
import random
import pandas as pd
import warnings
from scipy.signal import correlate2d
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

np.random.seed(1234)

## Parameters
N =  64
T = 200
c = 0.6 # 0.4   # 0.7
r = 1.4 # 0.5  # 0.7
p_loss = 0.5

# Analysis of the top/bottom Quantile
Quantile = 0.1


# In[2]:


test_import = ClusteringRateNeighbour_import(np.ones((2,2)))

test_import.mean()


# In[3]:


def Flattened_to_2DIndex(Array, n):
    '''
    Array: 1d enumeration of a previously n x n shaped array
    n: integer with len(Array) == n*n
    '''
    assert(len(Array) == n*n)
    i_index = (Array/n).astype(int)
    j_index = Array%n
    output = np.empty((n*n,2))
    output[:,0] = i_index
    output[:,1] = j_index
    return(output)



def Sample_Agent_B(N_agents, Indices_A, CircularBoundary = True):
    '''
    N_agents: integer (number of agents)
    Indices_A: 2d integer array (kth row of this array gives the x/y coordinates of agent k)
    '''
    
    # create 2d enumeration and flatten it
    Enumeration = np.arange(0,N_agents*N_agents).reshape((N_agents,N_agents))
    FlatEnumeration = Enumeration.flatten()
    
    # transform the flattened enumeration to (i,j) indices of the 2d map
    indices = Flattened_to_2DIndex(FlatEnumeration, N_agents)

    # Do we need to take "periodic" boundary conditions into account?
    if CircularBoundary: # top and bottom and left/right margin are neighbours
        ListIndices_Neighbours_A = [( Indices_A + [1,0])%N_agents,
                                        ( Indices_A + [-1,0])%N_agents,
                                        ( Indices_A + [0,1])%N_agents,
                                        ( Indices_A + [0,-1])%N_agents
                                   ]
        
    else: # Calculate the L1 Distances from the Indices of A
        Norm  = (abs(indices - Indices_A)).sum(axis = 1)
        Norm.reshape((N_agents,N_agents))
    # Safety check
    #ax = sns.heatmap(Norm.reshape((N_agents,N_agents)), linewidth=0.5)
    #plt.title("Distances")
    #plt.show()
    
    # Now create a list of only those indices that are exactly 1 distance away from A
        ListIndices_Neighbours_A = list(indices[Norm == 1])
    
    return(ListIndices_Neighbours_A)
    
def CorrlationLength2D(Array, thresh = 1./5., mean = False):

    # Compute the 2d Autocorrelation
    Autocorr = correlate2d(Array, Array, mode='full', boundary='wrap', fillvalue=0)
    
    # prepare the slices for the row and column with highest autocorr
    Maximum = np.max(Autocorr)
    N_1,N_2 = Array.shape # size of original array
    Index_Corr1, Index_Corr2 = np.unravel_index(Autocorr.argmax(),Autocorr.shape) # find maximum autocorr: centre

    Row_with_Maximum = Autocorr[Index_Corr1, # +/- ensures that size is equal to original Array
                                (Index_Corr2 - int(N_2/2)):(Index_Corr2 + int(N_2/2)) ]
    Column_with_Maximum = Autocorr[(Index_Corr1 - int(N_1/2)):(Index_Corr1 + int(N_1/2)),
                                  Index_Corr2]
    
    Corr_Length_Row = np.sum(Row_with_Maximum > thresh * Maximum)
    Corr_Length_Column = np.sum(Column_with_Maximum > thresh * Maximum)

    if mean:
        return(np.mean([Corr_Length_Row, Corr_Length_Column]))
    else:
        return([Corr_Length_Row, Corr_Length_Column])    


# In[4]:


Sample_Agent_B(N_agents = N, Indices_A = np.array([0,0]).astype(int))


# In[5]:


class NetworkSimulation:

    def __init__(self, num, T, r,c):
        assert type(num)==int # number of agents
        assert type(T) == int # number of time step iterations
        assert r >= 0  # relative gain in gamble
        assert c >= 0 and c < 1 # relative loss in gamble: must be between 0 and 100%
        self.N = num
        self.T = T
        self.r = r
        self.c = c
        self.AgentsTimeSeries = np.ones((self.N,self.N,self.T+1))
        self.Agents = np.ones((self.N, self.N))  # current state of the agents

    def simulate(self, history = False, quantile = 0.1):
        
        self.Quantile = quantile  # clustering of the top and bottom quantile of agents is calculated
        
        if history:  # Record the clustering coefficient for each time unit
            self.cluster_top_timeseries = list()
            self.cluster_bottom_timeseries = list()            
            
        for t in range(self.T): # iterate over each time unit
            for n in range(self.N * self.N):  # on average: each agent faces the risk once
                
                # A has the risk and wants to buy insurance: sample coordinates i and j       
                i_A = np.random.randint(0,self.N) 
                j_A = np.random.randint(0,self.N)


                # Wealth of A and associated cost/gain and maximum Fee
                w_A = self.Agents[i_A,j_A]
                G = self.r*w_A
                C = self.c*w_A
                F_max = w_A - ((w_A + G)**0.5) * ((w_A-C)**0.5)


                # Neighbourhood of A
                Neighbourhood = Sample_Agent_B(N_agents = self.N, 
                                               Indices_A = np.array([i_A,j_A]).astype(int))
                Neighbourhood_Offers = dict() # store the F_min offer of each neighbour
                Neighbourhood_Coordinates = dict() # store the coordinates/indices of neighbours
                Neighbours = 0 # enumeration of the neighbours for dict
                for Agent in Neighbourhood:
                    i_B, j_B = Agent.astype(int)
                    w_B = self.Agents[i_B, j_B]
                    F_min = -w_B  + 0.5*np.sqrt(4*w_B**2 + (G+C)**2 ) + (C-G)/2.
                    Neighbourhood_Coordinates[Neighbours] = Agent
                    Neighbourhood_Offers[Neighbours] = F_min
                    Neighbours += 1

                # Identify best offer:
                B = min(Neighbourhood_Offers, key=Neighbourhood_Offers.get)
                i_B, j_B = Neighbourhood_Coordinates[B].astype(int)
                w_B = self.Agents[i_B, j_B]
                F_min = -w_B  + 0.5*np.sqrt(4*w_B**2 + (G+C)**2 ) + (C-G)/2.


                # Now Gamble
                p = np.random.uniform(0,1)
                win = p>p_loss
                
                if (F_min >= F_max) or (w_B <= C): 
                    # B demands more than A is willing to pay:
                    # no contract
                    
                    if win:
                        self.Agents[i_A, j_A] *= (1+self.r)
                    else:
                        self.Agents[i_A,j_A] *= (1-self.c)
                        
                        
                elif F_min < F_max and (w_B > C):
                    # make a contract at midway fee
                    F = 0.5*(F_min + F_max) 
                    
                    self.Agents[i_A,j_A] -= F 
                    
                    if win:
                        self.Agents[i_B, j_B] += (F +  G)
                    else:
                        self.Agents[i_B, j_B] += (F-C)

            if history: # Compute Clustering    
                self.cluster_top_timeseries += [ClusteringRateNeighbour_import(self.Agents>np.quantile(self.Agents,1-self.Quantile)).mean()]
                self.cluster_bottom_timeseries += [ClusteringRateNeighbour_import(self.Agents<=np.quantile(self.Agents,self.Quantile)).mean()]

            self.AgentsTimeSeries[:,:,t+1] = self.Agents
        
        # Final Clustering after last time step
        self.ClustTop = ClusteringRateNeighbour_import(self.Agents>np.quantile(self.Agents,1-self.Quantile)).mean()
        self.ClustBottom = ClusteringRateNeighbour_import(self.Agents<=np.quantile(self.Agents,self.Quantile)).mean()

        # Correlation Length
        self.CorrLengthTop = CorrlationLength2D(1.0*(self.Agents > np.quantile(self.Agents,1-self.Quantile)), mean = True)
        self.CorrLengthBottom = CorrlationLength2D(1.0*(self.Agents <= np.quantile(self.Agents,self.Quantile)), mean = True)



# In[6]:


c_scan = np.arange(0.0,0.99,0.05)
r_scan = np.arange(0.0, 2.01, 0.05)
print("c",c_scan)
print("r",r_scan)

# Clustering
Results_Top = np.zeros((len(c_scan), len(r_scan)))
Results_Bottom = np.zeros((len(c_scan), len(r_scan)))

# 
CorrLengths_Top = np.zeros((len(c_scan), len(r_scan)))
CorrLengths_Bottom = np.zeros((len(c_scan), len(r_scan)))


DF_Clustering = pd.DataFrame({})

for c in c_scan:
    for r in r_scan:
        testsim = NetworkSimulation(num = N,T=T, r=r, c=c)
        testsim.simulate()
        print(c,r,np.round(testsim.ClustTop,3), np.round(testsim.ClustBottom,3))
        
        Results_Top[np.where(c_scan == c)[0][0], np.where(r_scan == r)[0][0]] = testsim.ClustTop
        Results_Bottom[np.where(c_scan == c)[0][0], np.where(r_scan == r)[0][0]] = testsim.ClustBottom
        
        CorrLengths_Top[np.where(c_scan == c)[0][0], np.where(r_scan == r)[0][0]] = testsim.CorrLengthTop
        CorrLengths_Bottom[np.where(c_scan == c)[0][0], np.where(r_scan == r)[0][0]] = testsim.CorrLengthBottom

        DF_Clustering = pd.concat([DF_Clustering, pd.DataFrame({"c": [c],
                                                               "r":[r],
                                                               "Clust_Top": [testsim.ClustTop],
                                                               "Clust_Bottom": [testsim.ClustBottom],
                                                               "CorrLength_Top":[testsim.CorrLengthTop],
                                                               "CorrLength_Bottom": [testsim.CorrLengthBottom]})],
                                  ignore_index=True)

        DF_Clustering.to_csv("Parameterscan_N="+str(N)+"_T="+str(T)+".csv", index=False)
        
np.save( "Results_Top_N="+str(N)+"_T="+str(T)+".npy",Results_Top)
np.save( "Results_Bottom_N="+str(N)+"_T="+str(T)+".npy",Results_Bottom)

np.save( "CorrLengths_Top_N="+str(N)+"_T="+str(T)+".npy",CorrLengths_Top)
np.save( "CorrLengths_Bottom_N="+str(N)+"_T="+str(T)+".npy",CorrLengths_Bottom)
np.save("c_scan_N="+str(N)+"_T="+str(T)+".npy", c_scan)
np.save("r_scan_N="+str(N)+"_T="+str(T)+".npy", r_scan)

