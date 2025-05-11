import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Create Model Class that inherits nn.Module

class Model(nn.Module):
    # Input layer (team comps) ->
    # hidden layer1 (n neurons) -->
    # Hidden layer ->
    # output(classify)
    def __init__(self, input_features=326, h1=128 , h2=64, output_features = 2): #change input features when converting to league champs. output features is because there are 3 classifications
        super().__init__()
        self.fc1 = nn.Linear(input_features, h1) #fc stands for "fully connected"
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

#pick see for random
# torch.manual_seed(41)

#create an instance of model
model = Model(input_features=1700) #model is the class we created above

db_path = os.path.join('instance', 'listings.db')
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM match;", conn)
conn.close()

print(df['team1Win'].value_counts())

champ_cols = [f'champ{i+1}' for i in range(10)]
X_raw = df[champ_cols]
y = df['team1Win'].astype(int)

# Flatten all champ columns into a single column for consistent encoding
flattened = X_raw.values.flatten().reshape(-1, 1)

# Fit encoder on all champs across all slots
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(flattened)

# Encode each champ column separately using the same encoder, then horizontally stack them
encoded_columns = [encoder.transform(X_raw[[col]]).toarray() for col in X_raw.columns]
X_encoded = np.hstack(encoded_columns)
print(X_encoded.shape)
print("^bing")
# #Train Test Split! Set X, y
# X = df.drop('variety', axis=1)
# y = df['variety']

# Convert to numpy arrays
X = X_encoded
y = y.values

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)#, random_state=41)

# Convert X features to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y labels to tensors long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set the criterion of model to measure the error, how far off predictions are from the data
criterion = nn.CrossEntropyLoss()

# Choose Adam Optimizer, lr = learning rate(if error doesnt go down as we learn after a while we want to lower lr)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #lr change this will change how long it takes to learn

# Train model
# Epochs (one run through all the training data in our network)
n_epochs = 100
losses = []

for i in range(n_epochs):
    # Go forwards and get a prediction
    y_pred = model.forward(X_train) # Get predicted results

    # Measure the loss/error, gonna be high at first
    loss = criterion(y_pred, y_train) #pred vs train value

    # Keep track of losses. not required but helpful to see
    losses.append(loss.detach().numpy())

    # print every 10 epochs
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')

    # Do some back propigation : take error rate of forward propagation and feed it back
    # through the network to fine tune the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# plt.plot(range(n_epochs), losses) #doesnt work in vscode. can probably google how to make it work
# plt.ylabel("loss")
# plt.xlabel("epoch")

# Evaluate model on test data
with torch.no_grad(): # turns off back prop while we are evaluating
    y_eval = model.forward(X_test) # X_test are features are from our test set and y_eval will be predictions
    loss = criterion(y_eval, y_test) # Find loss

print(loss)


correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        # Show the raw output, true value, and predicted winner
        #print(f'{i + 1}.)  {str(y_val)} \t True: {y_test[i]} \t Predicted: {y_val.argmax().item()}')

        # Check if prediction was correct
        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'We got {correct} correct out of {len(y_test)}!')
pred_percent = (correct/(len(y_test)))
print(f'Thats a {pred_percent}% correct rate')


# Test model with new flower
# new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])

# with torch.no_grad():
#     print(model(new_iris))