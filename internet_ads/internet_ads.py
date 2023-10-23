import pandas as pd
import numpy as np

# Read the dataset without column names
dataset = pd.read_csv('ad.data', header=None)

# convert the laset column to binary values - ad (1) nonad(0)
def convert_to_numeric(x):
    if x == 'ad.':
        return 1.0
    elif x == 'nonad.':
        return 0.0
    else:
        return x
    
dataset.iloc[:, -1] = dataset.iloc[:, -1].apply(convert_to_numeric)
dataset_to_array = np.array(dataset)

# Handle the missing data
for i in range(dataset_to_array.shape[0] - 1):  # Iterate over rows (excluding the last row)
    for j in range(dataset_to_array.shape[1] - 1):  # Iterate over columns (excluding the last column)
        if dataset_to_array[i][j] == '   ?' or dataset_to_array[i][j] == '     ?' or dataset_to_array[i][j] ==  '?':
            dataset_to_array[i][j] = np.nan


dataset_to_array = dataset_to_array.astype(float)

# Handle missing data for numeric columns
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(dataset_to_array)
dataset_to_array = imputer.transform(dataset_to_array)


# feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
dataset_to_array = sc.fit_transform(dataset_to_array)

# feature selction 
from sklearn.feature_selection import VarianceThreshold
threshold = np.mean(np.var(dataset_to_array, axis=0))
# Create thresholder 
thresholder = VarianceThreshold(threshold=threshold)
# Create high variance feature matrix
dataset_to_array = thresholder.fit_transform(dataset_to_array)

# splitting the data to train test and test set 
from sklearn.model_selection import train_test_split
training_set , test_set = train_test_split(dataset_to_array,test_size= 0.2 , random_state=42 )

# getting the max neurons in the first layer 
nb_neurons = training_set.shape[1]

# update the sets to torch Tenosrs 
import torch
training_set = torch.Tensor(training_set)
test_set= torch.Tensor(test_set)

# building the SAE

from torch import nn
class SAE(torch.nn.Module):
    def __init__(self , ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_neurons , 30 ) 
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 20)
        self.fc5 = nn.Linear(20, 30)
        self.fc6 = nn.Linear(30, nb_neurons)
        self.activation = nn.Sigmoid()
    def forward(self , x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.fc6(x)
        return x 

# defining the sae object
import torch.optim as optim
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# trining the sae 
from torch.autograd import Variable
nb_ads =training_set.shape[0]
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
  train_loss = 0
  s = 0.
  for id_ad in range(nb_ads):
    input = Variable(training_set[id_ad]).unsqueeze(0)
    target = input.clone()
    if torch.sum(target.data > 0) > 0:
      output = sae(input)
      target.require_grad = False
      output[target == 0] = 0
      loss = criterion(output, target)
      mean_corrector = nb_neurons/float(torch.sum(target.data > 0) + 1e-10)
      loss.backward()
      train_loss += np.sqrt(loss.data*mean_corrector)
      s += 1.
      optimizer.step()
  print('epoch: '+str(epoch)+'loss: '+ str(train_loss/s))
  
# test the sae 
test_loss = 0
s = 0.
for id_ad in range(test_set.shape[0]):
  input = Variable(training_set[id_ad]).unsqueeze(0) 
  target = Variable(test_set[id_ad]).unsqueeze(0)
  if torch.sum(target.data > 0) > 0:
    output = sae(input)
    target.require_grad = False
    output[target == 0] = 0
    loss = criterion(output, target)
    mean_corrector = nb_neurons/float(torch.sum(target.data > 0) + 1e-10)
    test_loss += np.sqrt(loss.data*mean_corrector)
    s += 1.
print('test loss: '+str(test_loss.item()/s))