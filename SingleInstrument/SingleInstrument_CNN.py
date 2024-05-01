import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
torch.cuda.empty_cache()

class CNN(nn.Module):
    def __init__(self,  kernel, imageX, imageY):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel)
        self.conv2 = nn.Conv2d(8, 16, kernel)
        self.conv3 = nn.Conv2d(16, 32, kernel)
        self.conv4 = nn.Conv2d(32, 64, kernel)
        self.fc1 = nn.Linear(64*(imageX-(kernel-1)*4)*(imageY-(kernel-1)*4), 256)  # Dense fully connected layer
        self.fc2 = nn.Linear(256, 11)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the datasets from the file
with open('data/datasets.pkl', 'rb') as f:
    train_set_tensor, validation_set_tensor, test_set_tensor = pickle.load(f)

kernel = 5
imageX = train_set_tensor[0][0][0].shape[0]
imageY = train_set_tensor[0][0][0].shape[1]

# Model configuration parameters
cfg = dict()
cfg['numEpoch'] = 500
cfg['learning_rate'] = .00001
cfg['batchSize'] = 64

# DataFrame to store training statistics
cnn_stats = pd.DataFrame(columns=('loss_train', 'loss_validation', 'accuracy_train', 'accuracy_validation'))

# DataLoaders for training and validation sets
myLoader_train = DataLoader(train_set_tensor, shuffle=True, batch_size=cfg['batchSize'])
myLoader_validation = DataLoader(validation_set_tensor, shuffle=False, batch_size=cfg['batchSize'])

# Number of images in the training and validation sets
N_train = len(train_set_tensor)
N_validation = len(validation_set_tensor)

# Number of mini-batches
nbr_miniBatch_train = len(myLoader_train)
nbr_miniBatch_validation = len(myLoader_validation)

# Instantiate the CNN model and move it to GPU if available
myModel = CNN(kernel = kernel, imageX = imageX, imageY = imageY)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myModel.to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(myModel.parameters(), lr=cfg['learning_rate'])
myLoss = nn.CrossEntropyLoss()

# DataLoaders for training and validation sets
myLoader_train = DataLoader(train_set_tensor, shuffle=True, batch_size=cfg['batchSize'])
myLoader_validation = DataLoader(validation_set_tensor, shuffle=False, batch_size=cfg['batchSize'])

best_accuracy = 0.0
# Training loop
print("Training Started")
for epoch in range(cfg['numEpoch']):
    start_time = time.time()  # Record start time of epoch
    # Training
    myModel.train()
    running_loss_train = 0.0
    correct_train = 0
    total_train = 0
    for i, (inputs, labels) in enumerate(myLoader_train):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = myModel(inputs)
        loss = myLoss(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss_train += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Validation
    myModel.eval()
    running_loss_val = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in myLoader_validation:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = myModel(inputs)
            loss = myLoss(outputs, labels)
            
            running_loss_val += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # Calculate statistics
    loss_train = running_loss_train / nbr_miniBatch_train
    loss_val = running_loss_val / nbr_miniBatch_validation
    accuracy_train = 100 * correct_train / N_train
    accuracy_val = 100 * correct_val / N_validation
    
    end_time = time.time()  # Record end time of epoch
    epoch_time = end_time - start_time  # Calculate epoch duration
    
    # Print statistics
    print(f"Epoch [{epoch+1}/{cfg['numEpoch']}]: Loss: {loss_train:.4f}/{loss_val:.4f}; Accuracy: {accuracy_train:.2f}%/{accuracy_val:.2f}%; Time: {epoch_time:.2f} seconds")
    
    # Store data in DataFrame
    cnn_stats.loc[len(cnn_stats)] = [loss_train, loss_val, accuracy_train, accuracy_val]

    # Save model if validation accuracy is improved
    if accuracy_val > best_accuracy:
        best_accuracy = accuracy_val
        best_model_state = myModel.state_dict()

    # Save the best model state
    torch.save(best_model_state, 'best_model.pth')

# Write the DataFrame to a CSV file
csv_filename = "cnn_stats.csv"
cnn_stats.to_csv(csv_filename, index=False)
print(f"Training statistics saved to {csv_filename}")

# Plotting loss vs. iteration
plt.figure(figsize=(10, 5))
plt.plot(cnn_stats['loss_train'], label='Train Loss')
plt.plot(cnn_stats['loss_validation'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()
plt.savefig('LossvsEpoch.png')

# Plotting accuracy vs. iteration
plt.figure(figsize=(10, 5))
plt.plot(cnn_stats['accuracy_train'], label='Train Accuracy')
plt.plot(cnn_stats['accuracy_validation'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.legend()
plt.savefig('AccuracyvsEpoch.png')

# DataLoader for test set
myLoader_test = DataLoader(test_set_tensor, shuffle=False, batch_size=cfg['batchSize'])

# Number of images in the test set
N_test = len(test_set_tensor)

# Number of mini-batches
nbr_miniBatch_test = len(myLoader_test)

# Evaluate the model on the test set
# Load the trained model
bestModel = CNN(kernel = kernel, imageX = imageX, imageY = imageY)
bestModel.load_state_dict(torch.load('best_model.pth'))
bestModel.to(device)
bestModel.eval()

correct_test = 0
total_test = 0
with torch.no_grad():
    for inputs, labels in myLoader_test:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        outputs = bestModel(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

# Calculate test accuracy
accuracy_test = 100 * correct_test / N_test
print(f"Test Accuracy: {accuracy_test:.2f}%")