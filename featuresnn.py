import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np 
import pandas as pd 
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# INPUT-HIDDEN_lAYER-OUTPUT DIMENTIONS
L_dim = dict()
L_dim['D_in'] = 58
L_dim['H_1'] = 256
L_dim['H_2'] = 128
L_dim['H_3'] = 64
L_dim['D_out'] = 10

# TRAINING PARAMETERS
hyperP = dict()
hyperP['numEpoch'] = 200
hyperP['learning_rate'] = .0001
hyperP['batchSize'] = 200

# READING DATA
data = pd.read_csv('features_30_sec.csv') 

# REMOVING FILE NAME COLUMN
data = data.drop(['filename'],axis=1)

# CHANGING LABEL COLUMN TO INTEGERS
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
# X = np.array(data.iloc[:, :-1], dtype = float)

# SCALING ALL THE FEATURES
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

# CONVERSION TO TENSORS
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# myModel = nn.Linear(58,10)
# myModel = nn.Sequential(*[nn.Linear(58,10), nn.ReLU(), nn.Linear(58,10)])

# SPLITTING THE DATA
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2)
train_data = data_utils.TensorDataset(X_train, y_train)
test_data = data_utils.TensorDataset(X_test, y_test)

N_train = len(train_data) # number of songs in the training set
N_test = len(test_data) # number of songs in the test set

# LOADING THE TENSORDATA
myLoader_train = data_utils.DataLoader(train_data, batch_size=hyperP['batchSize'], shuffle=True)
myLoader_test = data_utils.DataLoader(test_data, batch_size=hyperP['batchSize'], shuffle=True)

nbr_miniBatch_train = len(myLoader_train) # number of mini-batches
nbr_miniBatch_test = len(myLoader_test)

# MODEL DEFINITION
myModel = nn.Sequential(
    nn.Linear(L_dim['D_in'], L_dim['H_1']),
    nn.ReLU(),
    nn.Linear(L_dim['H_1'], L_dim['H_2']),
    nn.ReLU(),
    nn.Linear(L_dim['H_2'], L_dim['H_3']),
    nn.ReLU(),
    nn.Linear(L_dim['H_3'], L_dim['D_out']),
)

# CROSS ENTROPY LOSS
myLoss = nn.CrossEntropyLoss()
# OPTIMIZER
optimizer = torch.optim.Adam(myModel.parameters(), lr=hyperP['learning_rate'])

# TRAINING AND TESTING
t0 = time.time()
for epoch in range(hyperP['numEpoch']):
    # a new epoch begins
    print('-- epoch '+str(epoch))
    # Training
    running_loss_train = 0.0
    accuracy_train = 0.0
    myModel.train()
    for X,y in myLoader_train:
        # 1) initialize the gradient "∇ loss" to zero
        optimizer.zero_grad()
        # 2) compute the score and loss 
        score = myModel(X)
        loss = myLoss(score, y)
        # 3) estimate the gradient (back propagation)
        loss.backward()
        # 4) update the parameters
        optimizer.step()
        # 5) estimate the overall loss over the all training set
        running_loss_train += loss.detach().numpy()
        accuracy_train += (score.argmax(dim=1) == y).sum().numpy()
    loss_train = running_loss_train/nbr_miniBatch_train
    accuracy_train /= N_train
    print('     loss training: '+str(loss_train))
    print('     accuracy training: '+str(accuracy_train))
    # Testing
    running_loss_test = 0.0
    accuracy_test = 0.0
    myModel.eval()
    with torch.no_grad():
        for X,y in myLoader_test:
            # 1) compute the score and loss
            score = myModel(X)
            loss = myLoss(score, y)
            # 2) estimate the overall loss over the all test set
            running_loss_test += loss.detach().numpy()
            accuracy_test += (score.argmax(dim=1) == y).sum().numpy()
    loss_test = running_loss_test/nbr_miniBatch_test
    accuracy_test /= N_test
    print('     loss test: '+str(loss_test))
    print('     accuracy test: '+str(accuracy_test))
    # end epoch