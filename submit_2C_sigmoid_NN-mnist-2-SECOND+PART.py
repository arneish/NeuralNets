
# coding: utf-8

# In[66]:

import numpy as np
import pandas as pd
import math
import copy 
from random import shuffle

# df_train = pd.read_csv("./NN/toy_trainX.csv", header=None)
# df_test = pd.read_csv("./NN/toy_testX.csv", header=None)
# df_ltrain = pd.read_csv("./NN/toy_trainY.csv", header=None)
# df_ltest = pd.read_csv("./NN/toy_testY.csv", header=None)
# print(df_test.head())
# traind = df_train.values
# labeld = df_ltrain[0].values
# testd = df_test.values
# labeltest = df_ltest[0].values


# x_train = []
# y_train = []
# x_test = []
# y_test = []

traind = []
labeld = []
testd = []
labeltest = []

def getstuff(filename, tlist, llist):
    df=pd.read_csv(filename, sep=',',header=None)
    llist=list([df.values[i][-1] for i in range(len(df))])
    df = df *1.0/255.0
    totlist=list(df.values[i].tolist() for i in range(len(df)))
    tlist = [totlist[i][:-1] for i in range(len(totlist))]       
    return tlist, llist
traind, labeld=getstuff("/home/arneish/Desktop/ML_assgn3/NN/mnist_data/MNIST_train.csv", traind, labeld)
testd, labeltest=getstuff("/home/arneish/Desktop/ML_assgn3/NN/mnist_data/MNIST_test.csv", testd, labeltest)
traind = np.asarray(traind)
testd = np.asarray(testd)
labeld = np.asarray(labeld)
labeltest = np.asarray(labeltest)
#make 6 = 0
#make 8 = 1
labeld_f = []
labelt_f = []
for i in range (len(labeld)):
    if (labeld[i]==6):
        labeld_f.append(0)
    else:
        labeld_f.append(1)
        
for i in range (len(labeltest)):
    if (labeltest[i]==6):
        labelt_f.append(0)
    else:
        labelt_f.append(1)
labeld_f = np.asarray(labeld_f)
labelt_f = np.asarray(labelt_f)
labeld = labeld_f
labeltest = labelt_f

zero = 0
for i in range(len(labeld)):
    if (labeld[i]==0):
            zero+=1


# In[67]:

lenvalid = math.ceil(0.005*len(traind))
init = [i for i in range(len(traind))]
shuffle(init)
print(init[0:5])
validd = [traind[i] for i in init[0:lenvalid]]
labelvalid = [labeld[i] for i in init[0:lenvalid]]
traind2 = [traind[i] for i in init[lenvalid:len(init)]]
labeld2 = [labeld[i] for i in init[lenvalid:len(init)]]
print(len(labelvalid))


# In[114]:

#from last hidden layer to the decision layer 
#Sigmod[list]
def sigmoid(a):
    #print("sig b4r:", a)
    a = 1.0/(1.0+np.exp(-1*a))
    #print("sigmoid:", a)
    return a

#Prediction-datapoint
# def predict_data(X, wt_arch):
#     y_pred_vec = []
#     for x in X:
#        # print("predict data call", x)
#         y_pred = -1
#         output = list(x)
#         output.append(1.0)
#         print("len", len(wt_arch))
#         for layer in range(len(wt_arch)):
#                 w_matrix=wt_arch[layer]
#                 input = []
#                 input = list(output)            
#                 output = []
#                 for target in range(len(w_matrix[0])):
#                     output.append(sigmoid(sum(input*w_matrix[:,target])))
#                 output.append(1.0)
#         y_pred = output.index(max(output[:-1])) #0 or 1 
#         y_pred_vec.append(y_pred)
#     return np.asarray(y_pred_vec)


#Prediction
def predict(S, data, wt_arch): 
    #S: indices of input dataset 
    #print("predict is called")
    y_pred = []
    y_pred_vec = []
    for i in S:
        outputlayer = [] #length: #hidden-layers+1; index = 0 is first hidden layer
        output = list(data[i])
        output.append(1.0)
        #print(output)
        #print("len:",len(wt_arch))
        for var in range(len(wt_arch)):
            w_matrix=wt_arch[var]
            #print("layer, wmatrix:", layer, " ",w_matrix)
            input = []
            input = list(output)            
            output = []
            for target in range(len(w_matrix[0])):
                #print(sum(input*w_matrix[:,target]))
                output.append(sigmoid(sum(input*w_matrix[:,target])))
            output.append(1.0)
            #print("output", output)
            outputlayer.append(output)
        y_pred_vec.append(output)
        y_pred.append(output.index(max(output[:-1]))) #0 or 1 
    return y_pred, y_pred_vec, outputlayer

#Actual 
def actual(S, label):
    actual = []
    for i in S:
        actual.append(label[i])
    return actual

#Label-vector construction:
def labelvector(l):
        if (l==0):
            return [1,0]
        else:
            return [0,1]
            
#Backward pass

def backpass(i, data, label_vec, y_pred_vec, outputlayer, wt_arch, NN_arch, del_wt_arch):
    global dim_out
    global dim_in 
#     print("del wt arch size:", len(del_wt_arch))
    del_wt = np.zeros(shape = (NN_arch[-1]+1, dim_out))
    #for last layer 
#     print(del_wt)
    #decision-layer derivatives:
    del_layer = [0.0]*len(wt_arch) #deltas for #h + 1
    delta = [0.0]*(dim_out) 
    for j in range(dim_out): #each node of decision-layer
            output = outputlayer[-1]
#             print("output:", output)
            l = labelvector(label_vec)
#             print("label:", l)
            o = y_pred_vec[0]
#             print("y_pred", o)
            
            delta[j]=(l[j]-o[j])*o[j]*(1.0-o[j])
#             print("len wtarch[-1]:", len(wt_arch[-1]))
            for k in range(len(wt_arch[-1])):   
                del_wt[k][j]+=-1*outputlayer[-2][k]*delta[j]
#                 print("k", del_wt[k][j])
    del_wt_arch[len(del_wt_arch)-1]+=del_wt
    
    #hidden-layer derivatives from #h-1 to #0:
    for layer in range(len(NN_arch)-1,-1,-1):
        delta_down = delta[:] #list of deltas for down-layer nodes 
        delta = [] #list of deltas for current-layer nodes
        output = outputlayer[layer] #output of current layer 
        if (layer == 0):
            output_prev = list(data[i])
            output_prev.append(1.0)
        else:
            output_prev = outputlayer[layer-1] #output of previous layer
        w = wt_arch[layer+1]
        
        if (layer ==0):
                N = dim_in+1
        else:
                N = NN_arch[layer-1]+1
        
        del_wt = np.zeros(shape=(N, NN_arch[layer])) #delta weight matrix 
        
        for j in range(NN_arch[layer]):#for each node in layer            
            sigma = 0.0
            #print("layer, len deltadown ", layer, " ",len(delta_down))
            for l in range(len(delta_down)):
                sigma+=delta_down[l]*w[j][l]*output[j]*(1.0-output[j])
            delta.append(sigma) #add it to list of deltas for the layer nodes
            
            for k in range(N):
                #print("k:", k)
#                 print("delwt", del_wt)
                del_wt[k][j]=-1*sigma*output_prev[k]
        
        del_wt_arch[layer]+= del_wt

##############################################################################################################
#Weight Updates

def weight_update(counter, NN_arch, wt_arch, del_wt_arch, eta, batchsize):
    print("weight update eta:", eta/(math.sqrt(counter)))
    for layer in range (len(wt_arch)):
        #0 to h 
        wt_arch[layer]=wt_arch[layer] - eta/(math.sqrt(counter)*batchsize)*del_wt_arch[layer]
    return wt_arch
        

testacc  = []
trainacc = []
valacc = []
#Backpropagation





def Backpropagation (data, label, batchsize, NN_arch, wt_arch, eta, maxepoch):
    global testacc
    global trainacc
    global valacc
    global testd
    global labeltest
    global eps
    testacc= []
    trainacc = []
    valacc = []
    numepoch = 0
    #print("bp:", len(wt_arch))
    counter = 0
    counter_weight = 0
    while ((err_convergence(eps)==False) and (numepoch<maxepoch)):
        #construct one epoch run
        counter+=1
        print("counter:",counter)
        numepoch+=1
        init = [i for i in range (len(data))]
        shuffle(init)
        counter_in = 0
            
        for i in range(0, len(init), batchsize):
            counter_weight+=1   
            counter_in+=1
            print("counter_in",counter_in)
            if (i+batchsize>len(init)):
                break
            S = [init[j] for j in range(i, i+batchsize)]
            assert(len(S)==batchsize)
            
            #call FP, BP for all elements of this mini-batch
            
            del_wt_arch = [0]*len(wt_arch)
            #for every layer 
            for i in S:
                _, y_pred_vec, outputlayer_bp = predict([i], data, wt_arch)
                
                #print("y_pred_vec", y_pred_vec)
#                 print("i", i)
                backpass(i, data, label[i], y_pred_vec, outputlayer_bp, wt_arch, NN_arch, del_wt_arch)
            #call weight update after a stochastic pass through entire data set 
            
            weight_update(counter_weight, NN_arch, wt_arch, del_wt_arch, eta, batchsize)
            if (counter_in%10==0):
#                 acc =Test(validd, labelvalid, wt_arch)  
#                 valacc.append(acc)
                acc = Test(testd, labeltest, wt_arch)
                print("testaccuracy:", acc)
                testacc.append(acc)
                acc = Test(traind2, labeld2, wt_arch)  
                print("train acc:", acc)
                trainacc.append(acc)
            if (err_convergence(eps)==True):
                    break;
            #one epoch finished
       
#         acc = Test(validd, labelvalid, wt_arch)  
#         valacc.append(acc)
#         acc = Test(testd, labeltest, wt_arch)
#         print("testaccuracy:", acc)
#         testacc.append(acc)
#         #print("train acc:")
#         acc = Test(traind, labeld, wt_arch)  
#         trainacc.append(acc)
#     global wt_arch_save
#     if (convergence(valacc, wt_arch)==True):
#          wt_arch = wt_arch_save
    
    
    
#Testing:
#test=[]

result = []
wt_arch = []
error_list = []
    
def Test(data, label, wt_arch):
    #1 : 
    
   # global test
    global error_list
    global result
    global result_1
    global result_0
    S = [i for i in range(len(data))]
    correct = 0
    wrong = 0
    y_pred,y_pred_vec,_ = predict(S, data, wt_arch)
    
    
    errorsum = 0
    
    for i in S:
        if (len(data)==len(traind2)):
            error = np.linalg.norm(np.asarray(labelvector(label[i]))-np.asarray(y_pred_vec[i][0:2]))
            error = error**2
            errorsum+=error
        result.append(y_pred[i])
        #print("test: pred vs actual", y_pred[i], "/", label[i])
        if (y_pred[i]==label[i]):
            correct+=1
        else:
            wrong+=1
    #print(correct/(correct+wrong)*100 )
    if (len(data)==len(traind2)):
        error_list.append(errorsum)
    print("errorsum", errorsum)
    #test.append((correct/(correct+wrong))*100)
    return (correct/(correct+wrong))*100


def err_convergence(eps):
    global error_list
    if (len(error_list)<2):
        return False
    else:
        avg_1 = (error_list[-1])
        avg_2 = (error_list[-2])
        abs_err = math.fabs(avg_1-avg_2)
        rel_err = math.fabs((avg_1 - avg_2)/(avg_2)) 
        print("rel error:",rel_err)#math.fabs(error_list[-2]-error_list[-1])/math.fabs(error_list[-1]) )
        print("err list -1", error_list[-1])
        if (rel_err<eps):
            print("error convergence hit:",error_list[-1], math.fabs(error_list[-2]-error_list[-1]))
            return True
        else:
            return False
        

dec = 0
def convergence(valacc, wt_arch):
    global dec
    global max_dec
    global wt_arch_save
    if (len(valacc)==1 or len(valacc)==0):
        return False
    if (valacc[-1]<valacc[-2]):
        dec +=1
        print("dec:", dec)
        if (dec==1):
            wt_arch_save = copy.deepcopy(wt_arch)
    else:
        dec = 0
    if (dec == max_dec):
        print("CONVERGENCE HIT")
        return True
    else:
        return False


dim_in = 784 ##dimensions of input data 
dim_out = 2 #size of decision class [#nodes in decision layer]
NN_arch = [100] #the NN-architecture list
wt_arch = [] #list of weight matrices from one layer to next #size = h+1
eta = 0.1 #initialisation value for all weights
eps = 1
batchsize = 100 #SGD Batch size
max_dec = 5 #maximum number of decrements for SGD stopping

w=np.random.normal(0,1.0,(dim_in+1,NN_arch[0]))
wt_arch.append(w) 
#for the layer from Input to first hidden layer
for i in range(len(NN_arch)-1):
    
    #w = np.empty(shape=(NN_arch[i]+1, NN_arch[i+1]))
#     w = np.random.normal(0,1/math.sqrt(NN_arch[i]),(NN_arch[i]+1, NN_arch[i+1]))
    w = np.random.normal(0,1.0,(NN_arch[i]+1, NN_arch[i+1]))
    wt_arch.append(w)
# w = np.random.normal(0,1/math.sqrt(NN_arch[-1]),(NN_arch[-1]+1, dim_out))
w = np.random.normal(0,1.0,(NN_arch[-1]+1, dim_out))
wt_arch.append(w)
#wt_arch.append(np.empty(shape = (NN_arch[-1]+1, dim_out)))
print(len(wt_arch))


# In[ ]:

eps = 0.01
import time
start_time = time.time()
Backpropagation(traind2, labeld2,100, NN_arch, wt_arch, 20, 20) 
print("--- %s seconds ---" % (time.time() - start_time))


# In[109]:




# In[ ]:

# maxep =500
# nn = 2
# et = 200


# In[112]:

import math
import matplotlib.pyplot as plt
print(len(testacc))
x = np.arange(0, len(testacc))
y = [testacc[i] for i in x]
y2 = [trainacc[i] for i in x]
#y_val = [valacc[i] for i in x]
plt.figure(figsize=(10,10))
plt.plot(x, y, 'r', label = 'test')
plt.plot(x, y2, 'b', label = 'train')
#plt.plot(x, y_val, 'g', label = 'valid')
#plt.plot(x, trainacc, 'b')
plt.legend()
plt.savefig('mnnnmnist_5nodes.png')
plt.show()


# In[113]:

print(testacc)
#print((valacc[-1]))
print(trainacc)
print(error_list)
for i in range(len(error_list)-1):
    print(error_list[i]-error_list[i+1])


# In[ ]:

print(testacc[-1])
print(trainacc[-1])

X = np.asarray(traind)
y = np.asarray(labeld)
X_test = np.asarray(testd)
y_test = np.asarray(labeltest)
# X=np.random.rand(4,2)



import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(model, X, y):
    """
    Given a model(a function) and a set of points(X), corresponding labels(y), scatter the points in X with color coding
    according to y. Also use the model to predict the label at grid points to get the region for each label, and thus the 
    descion boundary.
    Example usage:
    say we have a function predict(x,other params) which makes 0/1 prediction for point x and we want to plot
    train set then call as:
    plot_decision_boundary(lambda x:predict(x,other params),X_train,Y_train)
    params(3): 
        model : a function which expectes the point to make 0/1 label prediction
        X : a (mx2) numpy array with the points
        y : a (mx1) numpy array with labels
    outputs(None)
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.05
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

plot_decision_boundary(lambda x:predict_data(x, wt_arch), X_test, y_test)
plt.savefig("3_eta40.png")
plt.show()
# Test(traind, labeld, wt_arch)


# In[102]:




# In[ ]:

# label = [0,1,2,3]
# labelvector(label)


# In[ ]:

# data = [[1.23, 2.354]]
# label =[0]
# y_pred,y_pred_v,opl= predict([0], data, wt_arch)
# print("-----------------")
# print(opl)
# del_wt_arch = []
# del_wt_arch = [0]*len(wt_arch)
# print(del_wt_arch)
# # for i in range (len(del_wt_arch)):
# #     del_wt_arch[i] = []
# backpass(0, data, label[0], y_pred_v, opl, wt_arch, NN_arch, del_wt_arch)
# print(wt_arch[2])
# print("----------")
# print(del_wt_arch[2])
# print("eta", eta)
# weight_update(NN_arch, wt_arch, del_wt_arch, eta, batchsize)
# print(wt_arch[2])


# In[ ]:




# In[ ]:

#data = [[1.23, 2.354]]
# label =[[0]]
   

