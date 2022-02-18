# Architecture of Main Model
"""
# Input = 224*224*3
# Conv1 = 11*11*3, stride = 4, inchannel = 3, outchannel = 96 | In original implementation, outchannel = 48 each for two GPUs
# Normalization(Batch) and Pool(Max), after first convolution
# Conv2 = 5*5*96, inchannel =96 , outchannel = 256| In original implementation, outchannel = 128 each for two GPUs
# Normalization(Batch) and Pool(Max), after second convolution
# Conv3 = 3*3*256, inchannel = 256 , outchannel = 384 | In original implementation, outchannel = 192 each for two GPUs
#Conv4 = 3*3*384, inchannel = 384, outchannel = 384| In original implementation, outchannel = 192 each for two GPUs
#Conv5 = 3*3*384, inchannel = 384 , outchannel = 256 | In original implementation, outchannel = 128 each for two GPUs
"""

#Some Details about the paper and implementation
''' 
   - There is no mention of padding in the original paper in the first layer,
   but without padding the the feature map height and width doesn't match to 55
   
   - AlexNet paper was one of the first implementation to use overlapping pooling and Dropout

   - In this implementation, sparse connection in layers 3,4 and 5 are replaced by dense connection. In original implementation, these layers were split into two GPUs, which is not done here.

   - The Data-Augmentation technique is not applied as mentioned in the paper. The imagenet dataset has been straight used.
   
   - Local Response used in this architecture to encourage lateral inhibition. In neurobiology, it means the capacity of a neuron to reduce the
   activity of its neighbor

   - Final layer in the model is a 1000-way softmax, but since pytorch combines nn.LogSoftmax() and nn.NLLLoss() in one single class for nn.CrossEntropy, softmax layer is not explicity
     encoded in the model

   - batch_size = 128, momentum = 0.9 and weight_decay = 0.0005

   - initialization : Weights - zero-mean Gaussian Distribution with standard deviation of 0.01
                    : Bias - 2,3,5th convolution layer and fully connected layer to be 1 and the remaining neuron biases 0
   
   - learning rate : Equal for all layers and lr was initialized = 0.01. Divide the learning rate by 10, when the validation error stopped at current learning rate. Don't have too many details 
    to use ReduceLronPlateau. So instead, used stepLRReduce on every 30 epoch. In the original paper, the LR was reduced 3 times by the factor of 10 throught the run of 90 epochs.
 
'''


from tabnanny import verbose
import config
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob
import numpy as np
import os
import datetime
import time
from PIL import Image


# Custom Dataloader
class CustomDataset(Dataset):
    def __init__(self,train=True,transform=None):
        self.data = []
        self.transform = transform 
        # Class to Label Encoding Loading
        class_to_label_loading = np.load("/afs/crc.nd.edu/user/a/abhatta/model_implementation/AlexNet/class_to_label.npy",allow_pickle=True)
        self.class_to_label = class_to_label_loading.item()
        if not train:
            self.data = np.load("/afs/crc.nd.edu/user/a/abhatta/model_implementation/AlexNet/val_data.npy")
        else:
            self.data = np.load("/afs/crc.nd.edu/user/a/abhatta/model_implementation/AlexNet/train_data.npy")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path)
        rgb_img = img.convert('RGB')
        label = self.class_to_label[class_name]
        if self.transform is not None:
            img_tensor = self.transform(rgb_img)
        label_tensor = torch.tensor([label])
        return img_tensor, label_tensor

# Model Definition:
class AlexNet(nn.Module):
   def __init__(self):
      super().__init__()
      # Defining convolution layers
      self.conv1 = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=2) 
      self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2) 
      self.conv3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1)
      self.conv4 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1)
      self.conv5 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1)
      # Defining Linear Layers
      self.fc1 = nn.Linear(13*13*256,4096)
      self.fc2 = nn.Linear(4096,4096)
      self.fc3 = nn.Linear(4096,1000)
      # Defining other utilities layers
      self.ReLU = nn.ReLU()
      self.maxPool= nn.MaxPool2d(kernel_size=3,stride=2)
      self.localNorm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
      self.Dropout = nn.Dropout(p=0.5)

   def forward(self,x):
      # Convolution Layers
      # Layer 1
      x = F.relu(self.conv1(x))
      x = F.local_response_norm(x,size=5, alpha=0.0001, beta=0.75, k=2)
      x = F.max_pool2d(x,kernel_size=3,stride=2)
      # Layer 2
      x = F.relu(self.conv2(x))
      x = F.local_response_norm(x,size=5, alpha=0.0001, beta=0.75, k=2)
      x = F.max_pool2d(x,kernel_size=3,stride=2)
      # Layer 3
      x = F.relu(self.conv3(x))
      # Layer 4
      x = F.relu(self.conv4(x))
      # Layer 5
      x = F.relu(self.conv5(x))
      # Dropout and Linear Layers
      x = F.dropout(x,p=0.5)
      x = x.view(x.size(0), -1) # Have to create a flat view inorder to pass it to the linear layer

      x = F.relu(self.fc1(x))
      x = F.dropout(x)
      x = F.relu(self.fc2(x))

      return self.fc3(x)


# Initialize weight and bias
def initialize_weights(m):
   for layer_count,layers in enumerate(m.children()):
      if isinstance(layers, nn.Conv2d):
         nn.init.normal_(layers.weight, mean=0, std=0.01)
         nn.init.constant_(layers.bias, 0) 
      if layer_count in [1,3,4,5,6]:
         nn.init.constant_(layers.bias, 1)

if __name__ == "__main__":

   #transform for Imagenet
   transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

   # Defining transform on the images
   #transform = transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop(224),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

   
   # Path to train and test set
   traindir = '/afs/crc.nd.edu/user/a/abhatta/ImagenetData/ILSVRC/Data/CLS-LOC/train/'
   
   print("Train DataLoading Started!")
   start = time.time()
   # Loading the training data
   train_dataset = CustomDataset(transform=transform,train=True)
   trainloader = DataLoader(train_dataset, batch_size=config.batch_size,
                                          shuffle=True, num_workers=8,pin_memory=True)
   
   print("Train DataLoading Ended!")
   end = time.time()
   print("Total time taken for Train Data Loading:",end-start)

   # check if GPU is available
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

   # Instantiation of the model class
   model = AlexNet()
   model.to(device)

   # Weight and Bias Initialization
   initialize_weights(model)

   #loss function and optimizers instantiation
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum,weight_decay=config.weight_decay)
   scheduler =  optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=0.1,verbose=True)

   #Training loop
   for epochs in range(config.epochs):
      model.train()
      training_loss = 0
      correct_pred_train = 0
      print("EPOCH Training {} Started!".format(epochs+1))
      for item,(data,labels) in enumerate(trainloader):
         (data,labels) = (data.to(device),labels.to(device))

         # print("Label Shape:",labels.shape)
         # print("Labels:",labels)

         optimizer.zero_grad()

         outputs = model(data)

         #print("Outputs",outputs)
         loss = criterion(outputs,labels.squeeze(1))
         loss.backward()
         optimizer.step()

         training_loss += loss.item()
         values,pred = torch.max(outputs.data, dim = 1)
         correct_pred_train += (pred == labels).sum().item() #torch.sum(pred==labels.squeeze(1))

      scheduler.step()
      print("Loss at the end of Epoch {} = {}".format(epochs+1,training_loss))
      print("Accuracy at the end of Epoch {} = {}". format(epochs+1,(correct_pred_train/len(train_dataset)*100)))

   # #Testing Loop
   # correct_pred_test = 0
   # with torch.no_grad():
   #    # set the model in evaluation mode, so batchnorm and dropout layers work in eval mode
   #    model.eval()
   #    # loop over the validation set
   #    for data, labels in testloader:
   #       (data, labels) = (data.to(device), labels.to(device))
   #       outputs = model(data)
   #       values,pred = torch.max(outputs, 1)
   #       correct_pred_test += torch.sum(pred==labels)
      
   #    print("Accuracy for the test set = {}". format(correct_pred_test/len(test_dataset)*100))

   # #Saving the model
   # torch.save(model.state_dict(),"alexnet_model.pth")



   


   





       
      

