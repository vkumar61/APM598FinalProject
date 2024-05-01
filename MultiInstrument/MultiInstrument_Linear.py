import pickle
import torch
import torch.nn as nn
import pandas as pd
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def myLoss(predicted_scores, true_values):
    # Apply sigmoid to predicted scores if not already sigmoided
    predicted_scores_sigmoid = torch.sigmoid(predicted_scores)

    # Calculate pointwise cross-entropy
    loss = -true_values * torch.log(predicted_scores_sigmoid + 1e-15) \
           - (1 - true_values) * torch.log(1 - predicted_scores_sigmoid + 1e-15)
    
    # Sum the losses across the 11 dimensions (instruments)
    loss = torch.sum(loss, dim=1)  # Sum along the second dimension

    # Compute the average loss across the batch
    loss = torch.mean(loss)  # Average across the batch dimension
    return loss

class LinearModel(nn.Module):
    def __init__(self, kernel, imageX, imageY):
        super(LinearModel, self).__init__()

        # Define layers
        self.pool = nn.MaxPool2d((kernel,kernel))  # Max pooling layer with kernel size 2x2 and stride 2
        self.fc = nn.Linear(int(imageX/(kernel)**2)*int(imageY/(kernel)**2), 11)  # Linear transformation from pooled/flattened input to output

    def forward(self, x):
        # Apply pooling twice
        x = self.pool(x)
        x = self.pool(x)
        # Apply linear transformation
        x = self.fc(x.view(x.size(0), -1))

        return x

# Load the datasets from the file
with open('data/multiDatasets.pkl', 'rb') as f:
    train_set_tensor, validation_set_tensor, test_set_tensor = pickle.load(f)

kernel = 2
imageX = train_set_tensor[0][0][0].shape[0]
imageY = train_set_tensor[0][0][0].shape[1]

cfg = dict()
cfg['numEpoch'] = 500
cfg['learning_rate'] = 0.00001
cfg['batchSize'] = 64
linear_stats = pd.DataFrame(columns=('loss_train', 'loss_validation','accuracy_train','accuracy_validation'))

myLoader_train = DataLoader(train_set_tensor, shuffle=True, batch_size=cfg['batchSize'])
myLoader_validation = DataLoader(validation_set_tensor, shuffle=False, batch_size=cfg['batchSize'])

N_train = len(train_set_tensor) # number of images in the training set
N_validation = len(validation_set_tensor)
nbr_miniBatch_train = len(myLoader_train) # number of mini-batches
nbr_miniBatch_validation = len(myLoader_validation)

myModel = LinearModel(kernel = kernel, imageX = imageX, imageY = imageY)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myModel.to(device)
optimizer = torch.optim.Adam(myModel.parameters(), lr=cfg['learning_rate'])

# Training loop
best_accuracy = 0.0
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
        predicted = torch.heaviside(torch.tanh(outputs.data), torch.tensor([0.0]))
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
            predicted = torch.heaviside(torch.tanh(outputs.data), torch.tensor([0.0]))
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # Calculate statistics
    loss_train = running_loss_train / nbr_miniBatch_train
    loss_val = running_loss_val / nbr_miniBatch_validation
    accuracy_train = 100 * correct_train / N_train / 11
    accuracy_val = 100 * correct_val / N_validation / 11
    
    end_time = time.time()  # Record end time of epoch
    epoch_time = end_time - start_time  # Calculate epoch duration
    
    # Print statistics
    print(f"Epoch [{epoch+1}/{cfg['numEpoch']}]: Loss: {loss_train:.4f}/{loss_val:.4f}; Accuracy: {accuracy_train:.2f}%/{accuracy_val:.2f}%; Time: {epoch_time:.2f} seconds")
    
    # Store data in DataFrame
    linear_stats.loc[len(linear_stats)] = [loss_train, loss_val, accuracy_train, accuracy_val]

    # Save model if validation accuracy is improved
    if accuracy_val > best_accuracy:
        best_accuracy = accuracy_val
        best_model_state = myModel.state_dict()

    # Save the best model state
    torch.save(best_model_state, 'best_model.pth')

# Write the DataFrame to a CSV file
csv_filename = "linear_stats.csv"
linear_stats.to_csv(csv_filename, index=False)
print(f"Training statistics saved to {csv_filename}")

# Plotting loss vs. iteration
plt.figure(figsize=(10, 5))
plt.plot(linear_stats['loss_train'], label='Train Loss')
plt.plot(linear_stats['loss_validation'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()
plt.savefig('LossvsEpoch.png')

# Plotting accuracy vs. iteration
plt.figure(figsize=(10, 5))
plt.plot(linear_stats['accuracy_train'], label='Train Accuracy')
plt.plot(linear_stats['accuracy_validation'], label='Validation Accuracy')
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
bestModel = LinearModel(kernel = kernel, imageX = imageX, imageY = imageY)
bestModel.load_state_dict(torch.load('best_model.pth'))
bestModel.eval()

correct_test = 0.0
total_test = 0.0
running_loss_test = 0.0
with torch.no_grad():
        for inputs, labels in myLoader_test:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = bestModel(inputs)
            loss = myLoss(outputs, labels)
            
            running_loss_test += loss.item()
            predicted = torch.heaviside(torch.tanh(outputs.data), torch.tensor([0.0]))
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

# Calculate test accuracy
accuracy_test = 100 * correct_test / N_test / 11
print(f"Test Accuracy: {accuracy_test:.2f}%")