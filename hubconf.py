def kali():
  print ('kali')
  
# sample invocation torch.hub.load(myrepo,'get_model',train_data=train_data,n_epochs=5, force_reload=True)
def get_model(train_data=None, n_epochs=10):
  model = None

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # set model variable to proper object, make use of train_data
  
  print ('Returning model... (rollnumber: xx)')
  
  return model
  
def test_model2(model, test_data):

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # set model variable to proper object, make use of train_data
  
  print ('Returning model... (rollnumber: xx)')
  
  return 1

# sample invocation torch.hub.load(myrepo,'test_model',model=model1,test_data=test_data,force_reload=True)
def test_model1(model=None, test_data=None):

  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # calculate accuracy, precision, recall and f1score
  
  print ('Returning metrics... (rollnumber: xx)')
  
  return accuracy_val, precision_val, recall_val, f1score_val


