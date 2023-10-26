import torch
import json
from torchvision import datasets, models, transforms
from torch import nn

import argparse

parser = argparse.ArgumentParser(description='Train a new network on a data set')

parser.add_argument('data_directory', type=str, help='Indicate data directory')
parser.add_argument('--save_dir', type=str, help='Set directory to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg', help='Choose architecture from vgg or densenet(will default to vgg16)')
parser.add_argument('--learning_rate', type=float, help='Set hyperparameter learning rate')
parser.add_argument('--hidden_units', type=int, help='Set hyperparameter hidden unit')
parser.add_argument('--epochs', type=int, help='Set hyperparameter epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

args = parser.parse_args()


# if gpu use set to true and gpu use is available
if args.gpu and torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU for training")
elif args.gpu and not torch.cuda.is_available():
    device = torch.device("cpu")
    print("GPU is not available, using CPU")
else:
    device = torch.device("cpu")
    print("Using CPU for training")

# set directories
data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# get hyperparameter values from command-line arguments or use defaults
learning_rate = args.learning_rate if args.learning_rate else 0.001
hidden_units = args.hidden_units if args.hidden_units else 4096
epochs = args.epochs if args.epochs else 3

# # hyperparameters dictionary
# hyperparameters = {
#    'learning_rate': learning_rate,
#    'classifier_linear': [25088, hidden_units],
#    'epochs': epochs
# }


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# TODO: Define your transforms for the training, validation, and testing sets
# make sure input data is resized to 224x224 pixels
# apply transformations such as random scaling, cropping, flipping
data_transforms = {
   'training' : transforms.Compose([
      transforms.RandomResizedCrop(224, scale = (0.8, 1.0)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
   ]),
   
   # no scaling or rotation transformations for validation and testing sets, resize and crip to size
   'validation' : transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
   ]),

   'testing' : transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
   ])
}

# TODO: Load the datasets with ImageFolder
image_datasets = {
   'training' : datasets.ImageFolder(train_dir, transform = data_transforms['training']),
   'validation' : datasets.ImageFolder(valid_dir, transform = data_transforms['validation']),
   'testing' : datasets.ImageFolder(test_dir, transform = data_transforms['testing'])
}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = [torch.utils.data.DataLoader(image_datasets['training'],
                                           batch_size = 64,
                                           shuffle = True),
               torch.utils.data.DataLoader(image_datasets['validation'], 
                                           batch_size = 64, 
                                           shuffle = True),
               torch.utils.data.DataLoader(image_datasets['testing'], 
                                           batch_size = 64, 
                                           shuffle = True)]


# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# set architecture using arch from script if provided, if not default to vgg
if args.arch == 'vgg':
   model = models.vgg16(pretrained=True)
   # freeze parameters
   for param in model.parameters():
      param.requires_grad = False
   classifier = nn.Sequential(
      nn.Linear(25088, hidden_units),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(hidden_units, 102),
      nn.LogSoftmax(dim=1)
   )
elif args.arch == 'densenet':
   model = models.densenet121(pretrained=True)
    
    # freeze parameters
   for param in model.parameters():
      param.requires_grad = False
   classifier = nn.Sequential(
       nn.Linear(1024, hidden_units),
       nn.ReLU(),
       nn.Dropout(0.5),
       nn.Linear(hidden_units, 102),
       nn.LogSoftmax(dim=1)
   )
else:
   print("Architecture input error, will default to vgg")
   model = models.vgg16(pretrained=True)
   # freeze parameters
   for param in model.parameters():
      param.requires_grad = False
   classifier = nn.Sequential(
      nn.Linear(25088, hidden_units),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(hidden_units, 102),
      nn.LogSoftmax(dim=1)
   )

# Commented out because classifier is now set on if condition statements
# classifier = nn.Sequential(
#    nn.Linear(25088, hidden_units),
#    nn.ReLU(),
#    nn.Dropout(0.5),
#    nn.Linear(hidden_units, 102),  
#    nn.LogSoftmax(dim=1)
# )

# resnet does not have classifer attacched
# model.fc = classifier
model.classifier = classifier


# loss function and optimizer
# Adam is a good default choice for learning speed accordingly
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr = learning_rate)

# accuracy
correct_predict = 0
sample = 0

# training
epoch_num = epochs
for epoch in range(epoch_num):
    # set in training
    model.train() 
    # used for later reporting for total loss during each epoch
    running_loss = 0

    for inputs, labels in dataloaders[0]:
        # reset to zero
        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device) # move to GPU if needed
        outputs = model(inputs) 

        # calculated by comparing predictions
        loss = criterion(outputs, labels)

        # backpropagation
        loss.backward()
        # update parameter
        optimizer.step()

        # keep track of loss
        running_loss += loss.item()

        #accuracy
        # torch.max get max and index of outputs for predictions and its label
        ignore, predict = torch.max(outputs, 1)
        # get number of correct predictions
        sample += labels.size(0)
        # add number of correct predictions
        correct_predict += (predict == labels).sum().item()

    accuracy = correct_predict / sample
    print(f"Epoch {epoch + 1}/{epoch_num}, Loss: {running_loss / len(dataloaders[0]):.4f}, Accuracy: {100 * accuracy:.2f}%")


# TODO: Do validation on the test set

# validation accuracy, validation loss
# set to evaluation mode
model.eval()

validation_loss = 0
correct_predict = 0
sample = 0

with torch.no_grad():
   for inputs, labels in dataloaders[1]:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)

      ignore, predict = torch.max(outputs, 1)
      sample += labels.size(0)
      correct_predict += (predict == labels).sum().item()

      # calculate validation loss
      loss = criterion(outputs, labels) 

      validation_loss += loss.item()
   
   validation_acc = correct_predict / sample
   validation_loss /= len(dataloaders[1])
   print(f"Validation Accuracy: {100 * validation_acc:.2f}%")
   print(f"Validation Loss: {validation_loss:.2f}")


# --save_dir argument
# TODO: Save the checkpoint 

# make path for checkpoint
checkpoint_path = 'checkpoint.pth'

class_to_idx = image_datasets['training'].class_to_idx

# save number of epochs, optimizer state
checkpoint = {
   'state_dict': model.state_dict(),
   'class_to_idx': class_to_idx,
   'epochs': epoch_num,
   'optimizer_state': optimizer.state_dict,
}

# save the created dict
torch.save(checkpoint, checkpoint_path)