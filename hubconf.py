# kali
import torch
from torch import nn
import torch.optim as optim

# You can import whatever standard packages are required

# full sklearn, full pytorch, pandas, matplotlib, numpy are all available
# Ideally you do not need to pip install any other packages!
# Avoid pip install requirement on the evaluation program side, if you use above packages and sub-packages of them, then that is fine!

###### PART 1 ######

# def get_data_blobs(n_points=100):
#   pass
#   # write your code here
#   # Refer to sklearn data sets
#   X, y = None
#   # write your code ...
#   return X,y

# def get_data_circles(n_points=100):
#   pass
#   # write your code here
#   # Refer to sklearn data sets
#   X, y = None
#   # write your code ...
#   return X,y

# def get_data_mnist():
#   pass
#   # write your code here
#   # Refer to sklearn data sets
#   X,y = None
#   # write your code ...
#   return X,y

# def build_kmeans(X=None,k=10):
#   pass
#   # k is a variable, calling function can give a different number
#   # Refer to sklearn KMeans method
#   km = None # this is the KMeans object
#   # write your code ...
#   return km

# def assign_kmeans(km=None,X=None):
#   pass
#   # For each of the points in X, assign one of the means
#   # refer to predict() function of the KMeans in sklearn
#   # write your code ...
#   ypred = None
#   return ypred

# def compare_clusterings(ypred_1=None,ypred_2=None):
#   pass
#   # refer to sklearn documentation for homogeneity, completeness and vscore
#   h,c,v = 0,0,0 # you need to write your code to find proper values
#   return h,c,v

# ###### PART 2 ######

# def build_lr_model(X=None, y=None):
#   pass
#   lr_model = None
#   # write your code...
#   # Build logistic regression, refer to sklearn
#   return lr_model

# def build_rf_model(X=None, y=None):
#   pass
#   rf_model = None
#   # write your code...
#   # Build Random Forest classifier, refer to sklearn
#   return rf_model

# def get_metrics(model1=None,X=None,y=None):
#   pass
#   # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
#   acc, prec, rec, f1, auc = 0,0,0,0,0
#   # write your code here...
#   return acc, prec, rec, f1, auc

# def get_paramgrid_lr():
#   # you need to return parameter grid dictionary for use in grid search cv
#   # penalty: l1 or l2
#   lr_param_grid = None
#   # refer to sklearn documentation on grid search and logistic regression
#   # write your code here...
#   return lr_param_grid

# def get_paramgrid_rf():
#   # you need to return parameter grid dictionary for use in grid search cv
#   # n_estimators: 1, 10, 100
#   # criterion: gini, entropy
#   # maximum depth: 1, 10, None  
#   rf_param_grid = None
#   # refer to sklearn documentation on grid search and random forest classifier
#   # write your code here...
#   return rf_param_grid

# def perform_gridsearch_cv_multimetric(model1=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  
#   # you need to invoke sklearn grid search cv function
#   # refer to sklearn documentation
#   # the cv parameter can change, ie number of folds  
  
#   # metrics = [] the evaluation program can change what metrics to choose
  
#   grid_search_cv = None
#   # create a grid search cv object
#   # fit the object on X and y input above
#   # write your code here...
  
#   # metric of choice will be asked here, refer to the-scoring-parameter-defining-model-evaluation-rules of sklearn documentation
  
#   # refer to cv_results_ dictonary
#   # return top 1 score for each of the metrics given, in the order given in metrics=... list
  
#   top1_scores = []
  
#   return top1_scores

# ###### PART 3 ######

# class MyNN(nn.Module):
#   def __init__(self,inp_dim=64,hid_dim=13,num_classes=10):
#     super(MyNN,self)
    
#     self.fc_encoder = None # write your code inp_dim to hid_dim mapper
#     self.fc_decoder = None # write your code hid_dim to inp_dim mapper
#     self.fc_classifier = None # write your code to map hid_dim to num_classes
    
#     self.relu = None #write your code - relu object
#     self.softmax = None #write your code - softmax object
    
#   def forward(self,x):
#     x = None # write your code - flatten x
#     x_enc = self.fc_encoder(x)
#     x_enc = self.relu(x_enc)
    
#     y_pred = self.fc_classifier(x_enc)
#     y_pred = self.softmax(y_pred)
    
#     x_dec = self.fc_decoder(x_enc)
    
#     return y_pred, x_dec
  
#   # This a multi component loss function - lc1 for class prediction loss and lc2 for auto-encoding loss
#   def loss_fn(self,x,yground,y_pred,xencdec):
    
#     # class prediction loss
#     # yground needs to be one hot encoded - write your code
#     lc1 = None # write your code for cross entropy between yground and y_pred, advised to use torch.mean()
    
#     # auto encoding loss
#     lc2 = torch.mean((x - xencdec)**2)
    
#     lval = lc1 + lc2
    
#     return lval
    
# def get_mynn(inp_dim=64,hid_dim=13,num_classes=10):
#   mynn = MyNN(inp_dim,hid_dim,num_classes)
#   mynn.double()
#   return mynn

# def get_mnist_tensor():
#   # download sklearn mnist
#   # convert to tensor
#   X, y = None, None
#   # write your code
#   return X,y

# def get_loss_on_single_point(mynn=None,x0,y0):
#   y_pred, xencdec = mynn(x0)
#   lossval = mynn.loss_fn(x0,y0,y_pred,xencdec)
#   # the lossval should have grad_fn attribute set
#   return lossval

# def train_combined_encdec_predictor(mynn=None,X,y, epochs=11):
#   # X, y are provided as tensor
#   # perform training on the entire data set (no batches etc.)
#   # for each epoch, update weights
  
#   optimizer = optim.SGD(mynn.parameters(), lr=0.01)
  
#   for i in range(epochs):
#     optimizer.zero_grad()
#     ypred, Xencdec = mynn(X)
#     lval = mynn.loss_fn(X,y,ypred,Xencdec)
#     lval.backward()
#     optimzer.step()
    
#   return mynn
def testHubCOnfig():
  return "It works"




  
  
  
  




