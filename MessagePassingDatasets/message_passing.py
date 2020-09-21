import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from torch_geometric.datasets import TUDataset

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import random
from sklearn.model_selection import ShuffleSplit
import os

from gp2_pytorch_graph import *


# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
#         self.lin = torch.nn.Linear(in_channels, out_channels)

#     def forward(self, x, edge_index):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]

#         # Step 1: Add self-loops to the adjacency matrix.
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#         # Step 2: Linearly transform node feature matrix.
#         x = self.lin(x)

#         # Step 3: Compute normalization.
#         row, col = edge_index
#         deg = degree(col, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         # Step 4-5: Start propagating messages.
#         return self.propagate(edge_index, x=x, norm=norm)

#     def message(self, x_j, norm):
#         # x_j has shape [E, out_channels]

#         # Step 4: Normalize node features.
#         return norm.view(-1, 1) * x_j



###                        Understanding their ENZYMES Dataset                        #### Uncomment me to see how I work

#x=torch.tensor([#nodes,#features per node], dtype=torch.float)
#edge_index=torch.tensor([pair 1], [pair 1 reverse], [pair 2], ... , [last pair reverse], dtype=torch.long)
#data = Data(x=x, edge_index=edge_index.t().contiguous())

# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
# datapoint=dataset[0]

##print(dataset[0]) #Data(edge_index=[2, 168], x=[37, 3], y=[1])
##print(datapoint.keys) #['x', 'edge_index', 'y']

# x=datapoint['x']
# edge_index=datapoint['edge_index']

#desired_shape1, desired_shape2=x[1],x[0]
# conv = GCNConv(3,37) #OR conv=GCNConv(x[1],x[0])
# x = conv(x, edge_index)
# targets_data=x
# df = pd.DataFrame(data=targets_data)

#df.to_csv (r'/Users/zachschmitz/Documents/GitHub/DevRep/Zach/graph_tutorial\export_dataframe.csv', index = False, header=True)


###                        Implementing My Dataset                        ####
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBIO
p = PDBParser(PERMISSIVE=1)
import numpy as np

from gp2_pytorch_graph import *
import torch
from torch_geometric.data import Data 

import csv
#I want to make a list of lists. Each list will have two elements.
#Those two elements will be the nodes connecting an edge if the distance 
#between those 2 nodes is below a certain cutoff which I will specify

#Have to make a list of the nodes that form an edge in reverse and forward direction
node_cxn_list=[]
#cutoff=5 #angstroms
cutoff=10 #angstroms
for i in range(0,len(CA)):
    #print('this is i ', i)
    for j in range(0,len(CA)):
        #print('this is j ', j)
        if adj[i][j]<=cutoff and i!=j:
            node_cxn_list.append([i,j])
#print(node_cxn_list) #print(adj)

edge_index=torch.tensor(node_cxn_list,dtype=torch.long)

# node_list=[[i] for i in range(0,len(CA))]
# heck_list=["heck" for i in range(0,len(CA))] #can't do this, doesn't like strings, has to be numbers/long/some torch datatype
# x=torch.tensor(node_list,fuck_list,dtype=torch.float) #doesn't work, need a single list of lists for the feature vector(s)

#node_hecklist=[[i, i+20] for i in range(0,len(CA))] #doing something like this gives me 2 feature vectors per node
# x=torch.tensor(node_hecklist,dtype=torch.float)
# data=Data(x=x,edge_index=edge_index.t().contiguous())

### Trying to define my feature vector by importing testfile.csv as a Pandas dataframe ###
#testfile_df=pd.read_csv("/Users/zachschmitz/Documents/GitHub/pytorch_geometric/August_Trial/MakeCSV/testfile.csv")

#print(testfile_df.iloc[0]) #gives the first row of the dataframe
#node_feature_list=[]
# for i in range(0,len(CA)):
    # node_feature_input=[i,testfile_df.lookup(0,'AA')
#x=torch.tensor(node_feature_list,dtype=torch.float)

#lookingup=testfile_df.lookup(1,'AA') #pulls an error, says that no rows name exist wtf
#fuckthis=testfile_df.get_value(0,"AA") #pulls an error no idea what is going on
#print(type(testfile_df)) #definitely a pandas dataframe
#finalcheck=testfile_df.loc[0,'AA']
#print(finalcheck) #this gives the full AA sequence of the (0th, "AA") entry 


###                       ***MANUALLY MAKING MY ENTIRE REFINED CSV***                        ###
import numpy as np
from functools import partial
import pandas as pd
pd.options.mode.chained_assignment = None
import os
import random
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from joblib import dump, load
import pickle

file="seq_to_assay_train_1,8,10.pkl"
df_original=pd.read_pickle(file)

avg_assay1=list((df_original['Sort1_1_score']+df_original['Sort1_2_score']+df_original['Sort1_3_score'])/3) #this finds the average of the 3 columns
avg_assay8=list((df_original['Sort8_1_score']+df_original['Sort8_2_score']+df_original['Sort8_3_score'])/3)
avg_assay10=list((df_original['Sort10_1_score']+df_original['Sort10_2_score']+df_original['Sort10_3_score'])/3)

#print(avg_assay1) #length 45433, dtype: float64
#NOTE: the sequences are various lengths due to gaps; I can't remember where they come from

def translate(dna):
    #print('entered')
    transdic={"TTT":"F","TTC":"F","TTA":"L","TTG":"L",
              "CTT":"L","CTC":"L","CTA":"L","CTG":"L",
              "ATT":"I","ATC":"I","ATA":"I","ATG":"M",
              "GTT":"V","GTC":"V","GTA":"V","GTG":"V",
              "TCT":"S","TCC":"S","TCA":"S","TCG":"S",
              "CCT":"P","CCC":"P","CCA":"P","CCG":"P",
              "ACT":"T","ACC":"T","ACA":"T","ACG":"T",
              "GCT":"A","GCC":"A","GCA":"A","GCG":"A",
              "TAT":"Y","TAC":"Y","TAA":"Z","TAG":"Z",
              "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
              "AAT":"N","AAC":"N","AAA":"K","AAG":"K",
              "GAT":"D","GAC":"D","GAA":"E","GAG":"E",
              "TGT":"C","TGC":"C","TGA":"Z","TGG":"W",
              "CGT":"R","CGC":"R","CGA":"R","CGG":"R",
              "AGT":"S","AGC":"S","AGA":"R","AGG":"R",
              "GGT":"G","GGC":"G","GGA":"G","GGG":"G"}
    AAseq=[]
    if len(dna)%3!=0:
        
        return "FRAMESHIFT"
    for i in range(0,len(dna),3):
        
        AAseq.append(transdic[str(dna[i:i+3])])
    AAseq=''.join(AAseq)
    if "KFWATV"==AAseq[0:6] and "VTRVRP"==AAseq[-6:] and "FEVPVYAETLDEALQLAEWQY" in AAseq: #FEVPIYAETLDEALELAEWQY is the original wildtype, this is the modified Kruziki version
        
        mid=AAseq.find("FEVPVYAETLDEALQLAEWQY")
        l1=AAseq[:mid]
        l2=AAseq[mid+len("FEVPVYAETLDEALQLAEWQY"):]
        if (6<=len(l1)<=8) and (6<=len(l2)<=8):
            if len(l1)==6:
                l1=l1[:3]+'XX'+l1[3:]
            elif len(l1)==7:
                l1=l1[:4]+'X'+l1[4:]
            if len(l2)==6:
                l2=l2[:3]+'XX'+l2[3:]
            elif len(l2)==7:
                l2=l2[:4]+'X'+l2[4:]
            AAseq=l1+l2
    else:
      return "Oh no"
    return AAseq

dna_seq=list(df_original["DNA"])
#dna_seq[0]=str(dna_seq[0])

for i in range(0,len(dna_seq)):
	dna_seq[i]=str(dna_seq[i])

aa_running=[]
for dna in dna_seq:
  aa_running.append(translate(dna))

#print(f'length of aa_running is {len(aa_running)}') #length is 45433, just as expected
for aa in aa_running:
  if aa=='Oh no':
    print("you have dna issues")

# print(type(dna_seq[0]))
# print(dna_seq[0])
# a=translate(dna_seq[0])
# print(f"a is {a} and b is {b}")
#aa_seq=[translate(dna_seq,0) for i in dna_seq]
#print(aa_seq)


##Setting up the pandas dataframe which I hope to eventually convert to csv/pkl
d={'Sort1_score':avg_assay1,'Sort8_score':avg_assay8,'Sort10_score':avg_assay10,"AA":aa_running}
df=pd.DataFrame(data=d,dtype=np.int8)


### ***Ok now trying to get the one-hot/ordinal to work out*** ###
#AAlist=np.array(list("ACDEFGHIKLMNPQRSTVWY")) #20 con. AA
AAlist=np.array(list("ACDEFGHIKLMNPQRSTVWXYZ")) #includes gap (X) and stop (Z)

def encode_everything_OH(enc,everything,axis):
    everything=np.array(list(everything))
    one_encode=enc.transform(everything.reshape(-1,1))
    return one_encode.flatten()

def encode_everything_ord(enc_ord,everything,axis):
    everything=np.array(list(everything))
    ord_encode=enc_ord.transform(everything.reshape(-1,1))
    return ord_encode.flatten()

enc_OH=preprocessing.OneHotEncoder(sparse=False)
enc_OH.fit(AAlist.reshape(-1,1))#unspecified number of rows, one column; this is just making a single feature column vector
encode_OH=partial(encode_everything_OH,enc_OH)
df.loc[:,'One_Hot']=df['AA'].apply(encode_OH,axis=1)

enc_ord=preprocessing.OrdinalEncoder()
enc_ord.fit(AAlist.reshape(-1,1))#unspecified number of rows, one column; this is just making a single feature column vector
encode_ord=partial(encode_everything_ord,enc_ord)
df.loc[:,'Ordinal']=df['AA'].apply(encode_ord,axis=1)

full_ordinal_list=[]
for index, row in df.iterrows():
  single_ordinal=[]
  single_ordinal=list(df.loc[index,'Ordinal'])
  full_ordinal_list.append(single_ordinal)

df.loc[:,"Ordinal"]=full_ordinal_list

print(type(df.loc[1,"Ordinal"])) #this is finally a LIST (not a string) whose individual elements are numpy.float64




### Making the expanded node feature list for all ~45,000 rows of the csv file based on Alex's data
node_feature_list=[]
for index, row in df.iterrows():
    
    single_graph=[]
    for i in range(0,len(CA)):
        node_feature_entry=[i,list(df.loc[index,'Ordinal'])[i]]
        #print(node_feature_entry)
        single_graph.append(node_feature_entry)
    node_feature_list.append(single_graph)
    #print(len(node_feature_list))
df['Feature']=node_feature_list


connectivity=[edge_index for i in range(0,len(df['Feature']))]
df['Connectivity']=connectivity

#x=df['Feature'][0]
#x=df.loc(0,'Feature')
#print(x)
#edge_index=df['Connectivity'][0]
#data=Data(x=x,edge_index=edge_index.t().contiguous()) ##This works, I can define each graph through indexing through "Feature" and "Connectivity" columns of the df
#print(data)


### Making my Pytorch Dataset ###
#for index, row in df.iterrows():




###     Write to my new CSV file   ###
df.to_csv(r'/Users/zachschmitz/Documents/GitHub/pytorch_geometric/August_Trial/AugustData/August25csv.csv', index = False, header=True)



###     Making the Dataset         ###
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import os.path as osp
import pandas as pd

class DevelopabilityDataset(InMemoryDataset): 

  root='/Users/zachschmitz/Documents/GitHub/pytorch_geometric/August_Trial' #where I want to save my dataset

  def __init__(self,root,transform=None,pre_transform=None):
    super(DevelopabilityDataset,self).__init__(root,transform,pre_transform)
    self.data,self.slices=torch.load(self.processed_paths[0])

  @property
  def raw_file_names(self):
    return []

  @property
  def processed_file_names(self):
    return ['/Users/zachschmitz/Documents/GitHub/pytorch_geometric/August_Trial/AugustData/August25csv.csv']

  def download(self):
    df=pd.read_csv(processed_file_names())
    return df

  def process(self):
    data_list=[]

    ##process by row index
    ##Initializing each graph Data object:
    for index, row in df.iterrows():
      x=df.loc[row,'Feature']
      edge_index=df.loc[row,'Connectivity']
      y=list(df.loc[row,'Sort1Score'],df.loc[row,'Sort8Score'],df.loc[row,'Sort10Score'])

      graph=Data(x=x,edge_index=edge_index.t().contiguous(),y=y)
      data_list.append(graph)

    print("I made it past the iteration")
    data,slices=self.collate(data_list)
    torch.save((data,slices),self.processed_paths[0])

a=DevelopabilityDataset(root='check')
print(a)

#for data in a:
#  print(data)

#print(df.iloc[0]) #gives the full first row of the pandas dataframe "df" here







#ordinalexpandedfile_df=pd.read_csv("/Users/zachschmitz/Documents/GitHub/pytorch_geometric/August_Trial/AugustData/wholeshebang.csv")
#print(f"The second row under Ordinal is {testfile_df.loc[1,'Ordinal']} and is of type {type(testfile_df.loc[1,'Ordinal'])}")

# index = testfile_df.index
# a_list = list(index)
# print(a_list) #gives me a list of row names from 0 to 45432 

#print(node_list)

#print(data.num_nodes) #45
#print(data.num_edges) #734
#print(data.num_node_features) #1
#print(data.contains_isolated_nodes()) #False
#print(data.is_directed) #False, "bound method"

#print(data.contains_self_loops()) #None
#print(data.num_faces) #None



# datapoint=data
# #print(my_graph)

# x=datapoint['x']
# edge_index=datapoint['edge_index']

# conv = GCNConv(1,45) #OR conv=GCNConv(x[1],x[0])
# x = conv(x, edge_index)
# targets_data=x
# df = pd.DataFrame(data=targets_data)


# #df.to_csv (r'/Users/zachschmitz/Documents/GitHub/pytorch_geometric/August_Trial/AugustData/singlegraph.csv', index = False, header=True)
# df.to_csv (r'/Users/zachschmitz/Documents/GitHub/pytorch_geometric/August_Trial/AugustData/manygraphs_takeone.csv', index = False, header=True)







