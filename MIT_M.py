import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import colorcet as cc
from matplotlib.colors import LogNorm, Normalize
from numpy import reshape

#numpy encoder for json export
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

#rotates matrix for export
def rotate_matrix(mat):
    n = len(mat)
    # Reverse the rows
    #mat.reverse()
    np.flip(mat)
    # Transpose the matrix
    for i in range(n):
        for j in range(i):
            mat[i][j], mat[j][i] = mat[j][i], mat[i][j]
    return mat

size = 100

#X_ORIGIN = np.logspace(3,6.2,num=size)
X_ORIGIN = np.logspace(3,6.1,num=size)
for i in range(len(X_ORIGIN)):
    X_ORIGIN[i] = (X_ORIGIN[i])-(10**3.0)

X1 = X_ORIGIN
X2 = X_ORIGIN
X3 = X_ORIGIN

ERN_LIB = {'Csy4':60.23, 
           'CasE':9.88,
           'Pgu':1.49}

ULIBTYPE = 'N3'

title_M = 'MIT M'

OUT = np.zeros(shape=(size,size,size))
n_A = np.zeros(shape=(size,size,size))
n_B = np.zeros(shape=(size,size,size))
n_C = np.zeros(shape=(size,size,size))
n_OUT = np.zeros(shape=(size,size,size))
n_OUT_slice = np.zeros(shape=(size,size))

UORF_LIB_N3 = {
'0x':1.0,
'1w':0.3710807074132212,
'1x':0.2181819749880083,
'2x':0.1535211541006999,
'3x':0.14844557079046325,
'4x':0.13388559629210156,
'5x':0.1303265828449993,
'6x':0.1271317612141076,
'8x':0.12664940091611943
}

UORFLIB = UORF_LIB_N3

picklepath_ratios_gt1 = 'interpolator_ratios_gt1VF.pkl'
picklepath_ratios_lt1 = 'interpolator_ratios_lt1VF.pkl'

with open(picklepath_ratios_gt1, 'rb') as f:  
    interpolator_ratios_gt1 = pickle.load(f)
with open(picklepath_ratios_lt1, 'rb') as f:  
    interpolator_ratios_lt1 = pickle.load(f)

#Interpolator Functions
IF = {
      'Rgt1':interpolator_ratios_gt1,
      'Rlt1':interpolator_ratios_lt1
      }

#RF = ratio function
#IF = the interpolation function library
#IN = the x coordinate (~valid within the range 10**3.0 to 10**6.2)
#ratio = the y coordinate (the ratio of part plasmid to marker plasmid)
#for ratios <1, all variables are normalized to [0,3.2]
#for ratios >1, all variables are normalized to [0,2]
#returns the MEF value of the component of interest at the IN value
def RF(IF,IN,ratio):
    if ratio > 1.0:
        intep = IF['Rgt1'] #for ratios greater than 1
        insert = np.array( [ [ (np.log10(IN)-3)*(2/3.2), (np.log10(ratio)) ] ] )
        temp = intep( insert )[0]
        return 10**(temp*(3.2/2.0)+3) 
    elif ratio < 1.0:
        intep = IF['Rlt1'] #for ratios less than 1
        insert = np.array( [ [ (np.log10(IN)-3), (ratio*3.2) ] ] )
        temp = intep( insert )[0] 
        return 10**(temp+3.0) 
    else: #if the ratio is 1
        return IN

#type is the ERN type of the current ERN function
#tlist is the type of intput into the ERN function
#IN is the value of the input that corresponds to the input type
#r is the list of all ratios
#u is the list of all uorfs
#otype is the type of the output (ERN or Fluorescent Protein)
def EF6(type,tlist,IN,r,u,otype='FP'): 
    output = 0.0
    ernsum = 0.0
    recsum = 0.0
    bias = 10**3.0
    for i in range(len(IN)):
        if tlist[i] == 'T-ERN': #carry through ERN
            ernsum+=IN[i]
        elif tlist[i] == 'T-REC': #carry through positive
            output+=IN[i] 
        elif tlist[i] == 'T-REC-N': #positive bias
            output+=UORFLIB[u[i]]*max(0, (RF(IF, (IN[i]+bias), r[i])-bias)) 
        elif tlist[i] == 'ERN': #ERN
            ernsum+=UORFLIB[u[i]]*max(0, (RF(IF, (IN[i]+bias), r[i])-bias)) 
        elif tlist[i] == 'REC': #REC
            recsum+=max(0, (RF(IF, (IN[i]+bias), r[i])-bias))
    raw_output = max(0, ((recsum) - ERN_LIB[type]*(ernsum))) 
    for i in range(len(IN)):
        if tlist[i] == 'REC': #if rec site
            output+=max(0,(UORFLIB[u[i]]*raw_output))
    return output

for i in range(size): #X1, EBFP2
    for j in range(size): #X2, MKO2
        for k in range(size): #X3 MMAROON1
            #M:
            #Csy4 node
            n_A[i][j][k] = EF6('Csy4', ['ERN', 'REC'], [X1[i], X3[k]], [1.0, 2.0], ['8x', '0x'], 'ERN') #output Pgu

            #M:
            #Pgu node
            n_B[i][j][k] = EF6('Pgu', ['ERN', 'T-ERN', 'REC'], [X1[i], n_A[i][j][k], X2[j]], [2.0, 1.0, .2], ['8x', '0x', '0x'], 'FP') #output is mNG

            #M:
            #CasE node
            n_C[i][j][k] = EF6('CasE', ['ERN', 'REC', 'T-REC-N', 'T-REC'], [X1[i], X3[k], X1[i], n_B[i][j][k]], [2.0, 1.0, 2.0, 1.0], ['0x', '0x', '4x', '0'], 'FP') #output is mNG

            n_OUT[i][j][k] = n_C[i][j][k]+10**3.0
            

for i in range(len(X_ORIGIN)):
    X_ORIGIN[i] = (X_ORIGIN[i])+(10**3.0)

X1 = X_ORIGIN
X2 = X_ORIGIN
X3 = X_ORIGIN

###########################################################################
#Create well spaced tick labels
spanfac = 8
logspaceout = np.log10(X1)
print("logspaceout")
print(logspaceout)
binmarksx,binmarksy = logspaceout, logspaceout
span = spanfac*(np.max(logspaceout)-np.min(logspaceout))/size
xticks = [('1e'+str(round(item, 1))) for item in binmarksy]
if len(xticks) > 5:
    div = round(size/3)
    for i in range(len(xticks)):
        if (i % div) > 0:
            xticks[i] = ''
yticks = xticks[::-1]
#########################################################################
#M
tempsumup = 0.0
tempcount = 0
for i2 in range(size):
    for j2 in range(size):
        for k2 in range(size):
            #match prediction bin to plot bin
            if (X3[k2] > 55715) and (X3[k2] < 147199):
                tempsumup += n_OUT[i2][j2][k2]
                tempcount += 1
        n_OUT_slice[i2][j2] = tempsumup/tempcount #avg of all that meet the z criteria
        tempsumup = 0.0
        tempcount = 0

n_OUT_invert = np.rot90(n_OUT_slice)

ticksize = 15
ylabel = "X\u2082 (MKO2)"
xlabel = "X\u2081 (EBFP2)"
cbartitle = 'Y (MNEONGREEN)'
SAVE = True

fig, ax = plt.subplots(figsize=(10,10))
heat_map = sns.heatmap(n_OUT_invert, 
                            linewidth=0, 
                            annot = False, 
                            xticklabels=xticks,
                            yticklabels=yticks,
                            cmap=cc.cm.linear_kbgyw_5_98_c62,
                            norm=LogNorm(),
                            cbar_kws={'label': cbartitle},
                            square=True
                            )


ax.set_title(title_M, fontsize = 28, pad=20)
plt.ylabel(ylabel, fontsize=24)
plt.xlabel(xlabel, fontsize=24)
plt.xticks(fontsize=ticksize, rotation=45)
plt.yticks(fontsize=ticksize, rotation=45)
cax = heat_map.figure.axes[-1]
cax.tick_params(labelsize=ticksize)
cax.yaxis.label.set_size(24)
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
if SAVE:
    save_path = (title_M)+".png"
    plt.savefig(save_path, dpi=600)
