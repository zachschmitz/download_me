from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBIO
p = PDBParser(PERMISSIVE=1)
import numpy as np

from adj1 import *
import torch
from torch_geometric.data import Data 


#I want to make a list of lists. Each list will have two elements.
#Those two elements will be the nodes connecting an edge if the distance 
#between those 2 nodes is below a certain cutoff which I will specify

#Have to make a list of the nodes that form an edge in reverse and forward direction
adj=weight

node_cxn_list=[]
#cutoff=5 #angstroms
cutoff=10 #angstroms
for i in range(0,len(CA)):
	#print('this is i ', i)
	for j in range(0,len(CA)):
		#print('this is j ', j)
		if adj[i][j]<=cutoff and i!=j:
			node_cxn_list.append([i,j])
#print(node_cxn_list)m#print(adj)

edge_index=torch.tensor(node_cxn_list,dtype=torch.long)

node_list=[[i] for i in range(0,len(CA))]
x=torch.tensor(node_list,dtype=torch.float)

data=Data(x=x,edge_index=edge_index.t().contiguous())

#print(node_list)

#print(data.num_nodes) #45
#print(data.num_edges) #734
#print(data.num_node_features) #1
#print(data.contains_isolated_nodes()) #False
#print(data.is_directed) #False, "bound method"

#print(data.contains_self_loops()) #None
#print(data.num_faces) #None

