import argparse

from torch import nn
import torch
from PIL import Image
from torchvision import transforms, models
import numpy as np
import json

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(path):
   #  load checkpoint
   checkpoint = torch.load(path)

   model = models.vgg16(pretrained=True)
   
   # rebuild the model
   classifier = nn.Sequential(
      nn.Linear(25088, 4096),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(checkpoint['hyperparameters']['classifier_linear'][0], checkpoint['hyperparameters']['classifier_linear'][1]),
      nn.LogSoftmax(dim=1)
   )

   model.classifier = classifier
   model.load_state_dict(checkpoint['state_dict'])
   model.class_to_idx = checkpoint['class_to_idx']

   return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    # transformation(same as training)
    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
    ])

    pil_image = transform(pil_image)

    # convert to numpy array
    np_image = np.array(pil_image)

    # pytorch expects the color channel to be the first dimension but it's the third dimension in the PIL and numpy array, reordered
   #  np_image = np_image.transpose((1, 2, 0))

    return np_image # imshow changes image to numpy so returing np_image creates error?

def predict(image_path, model, topk=5, category_names=None, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # set to evalution mode
    model.eval()

    # process_image
    image = process_image(image_path)

    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    probabilities = torch.exp(output)

    # get k probablities and indices of those probabilities corresponding to the classes
    top_probs, top_indices = probabilities.topk(topk, dim=1)
    # convert indices to actual class labels using class_to_idx
    # make sure to invert dictionary to get mapping from index to class
    idx_to_class = {i: key for key, i in model.class_to_idx.items()}
    classes = [idx_to_class[i.item()] for i in top_indices.squeeze()]

    probs = top_probs.squeeze().cpu().numpy().tolist()

    # check if category_names was wanted
    if category_names:
      with open(category_names, 'r') as f:
          cat_to_name = json.load(f, strict=False) # to avoid error at some workspaces and library versions, strict keyword must be set to false
      flower_names = [cat_to_name[cat] for cat in classes] # change class number to flower names
    else: # other, just return class number
      flower_names = classes

    return probs, flower_names

parser = argparse.ArgumentParser(description="Predict flower name from an image along with the probablity of the name")

# create command-line script for usage
parser.add_argument("image_path", type=str, help="Image path")
parser.add_argument("checkpoint", type=str, help="Checkpoint file path")
parser.add_argument("--top_k", type=int, default=5, help="Set number for top K most likely results")
parser.add_argument("--category_names", type=str, help="Use mapping of categories to real names")
parser.add_argument("--gpu", action="store_true", help="Using GPU")

# check command-line input
args = parser.parse_args()

# check GPU (can probably set if/check if gpu is available and/or checked for detailed output as in train.py)
device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

# load checkpoint for model
model = load_checkpoint(args.checkpoint)
# image_path = args.image_path
# image = process_image(image_path)

top_prob, top_class = predict(args.image_path, model, args.top_k, args.category_names, device)

# display results
for i in range(len(top_prob)):
    print(f"Class/Flower name: {top_class[i]}, Probablility: {100 * top_prob[i]:.2f}%")