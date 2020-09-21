#alpha carbon approach

#May 25, Zach Schmitz, walking through tutorial


###NOTE I can iterate over all atoms/residues in a given chain/residue/model etc. thorugh methods like
###"model.get_residues' or 'chain.get_atoms()''
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBIO
p = PDBParser(PERMISSIVE=1)
import numpy as np

###overal setup is structure/model/chain/residue/atom from most important->least important hierarchy
structure = p.get_structure('2wnm','2wnm.pdb')

###define model structure setup
model=structure[0] #there's only one structure, which makes sense as this is a crystallography file; nmr typically gives multiple structures
chain=model["A"] #there is only one peptide chain for gp2; there are 45 AAs in this chain #print(first_model)#print(chain_A)

###try to determine list of the residues that I can easily index into
residue=[]
for i in range(1,len(chain)+1):
	residue.append(chain[i]) #print(residue[0]) #<Residue LYS het= resseq=1 icode =>

###trying to find carbon atoms with for-loop over residues #pdb files label each atom with 4 chars
CA=[]
for i in range(0,len(chain)):
	CA.append(residue[i]['CA']) #print(len(CA)) #pos1=CA[0].get_vector();pos2=CA[1].get_vector();

                         ### MAKE THE ADJACENCY MATRIX ###
adj1=np.zeros((len(CA),len(CA)))
#print(np.size(adj)) #print(adj)

for i in range(0,len(CA)):
	atom1=CA[i]
	for j in range(0,len(CA)):
		atom2=CA[j]
		distance=atom1-atom2
		adj1[i][j]=distance #print(adj)

                          ### MAKE THE WEIGHT MATRIX ###
weight=np.zeros((len(CA),len(CA)))
for i in range(0,len(CA)):
	for j in range(0,len(CA)):
		if i!=j:
			weight[i][j]=1/adj1[i][j]
		else:
			weight[i][j]=0 #print(weight) 

                          ### HISTORGRAM PREPARARATION ###
#Just want a list, x1, that has all of the distances in a triangular matrix. 
x1=[]
for i in range(0,len(adj1)):
	for j in range(0,len(adj1)):
		if i>j:
			x1.append(adj1[i][j])


#                           ## HEATMAP PREPARATION ###

# import numpy as np #np.random.seed(0)
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.set()
# #ax = sns.heatmap(adj1, linewidths=.5, annot=True, fmt=".1f")
# ax = sns.heatmap(adj1, linewidths=.5, annot=False, fmt=".1f")
# ax.set_title("Adj1 Alpha Carbon Heatmap")

# #Trying to set x axis to the top
# ax.xaxis.tick_top()   
# ax.xaxis.set_label_position('top') 

# plt.show()




###Failed Matplotlib Attempt
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

# fig,ax=plt.subplots()
# im=ax.imshow(adj1)

# # We want to show all ticks...
# shape=adj1.shape
# ax.set_xticks(np.arange(shape[0]))
# ax.set_yticks(np.arange(shape[0]))

# # # ... and label them with the respective list entries
# # ax.set_xticklabels(adj1[0])
# # ax.set_yticklabels(adj1[0])

# # Create colorbar
# cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
# cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
# # # Rotate the tick labels and set their alignment.
# # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
# #          rotation_mode="anchor")

# ax.set_title("Adj1 Alpha Carbon Heatmap")
# fig.tight_layout()
# plt.show()












#print(x1)

#print(len(x1))
#print(len(CA))





#print(len(residue[0]))
#print(len(residue[1]))


#### MISCELLANEOUS NOTES/TESTS

#print(residue[0].is_disordered()) #returns 0 --> residue 0 (lysine) does NOT have disordered atoms

#residue1=chain[(' ', 1, ' ')]
#residue1_take2=chain[1]

#print(structure)

#ca1=residue['CA'];ca2=residue2['CA']
#istance=ca1-ca2;
#print(distance)


#resolution = structure.header['resolution']
#keywords=structure.header['keywords'] #keywords mysteriously doesn't work!


#io=PDBIO()
#io.set_structure(s) #it's pretty hilarious in their examples that "s" always brings up an error
#io.save('out.pdb')
