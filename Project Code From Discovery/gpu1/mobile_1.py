# Imports
import os
import random
import numpy as np
#import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torchvision import datasets, transforms, models 
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from natsort import natsorted
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")


# train dataset

train_dir = '/scratch/shetty.shra/Data/ASL_Alphabet_Dataset/asl_alphabet_train'
test_dir = '/scratch/shetty.shra/Data/ASL_Alphabet_Dataset/asl_alphabet_test'
Name = os.listdir(train_dir)
print(Name)
print(len(Name))

N=list(range(len(Name)))
normal_mapping=dict(zip(Name,N)) 
reverse_mapping=dict(zip(N,Name))

# 1 class
class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image        
        
### class 2
train_transform=transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(30),              # resize shortest side
        transforms.CenterCrop(30),          # crop longest side
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])

test_transform=transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(30),              # resize shortest side
        transforms.CenterCrop(30),          # crop longest side
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])


trainset = ImageFolder(train_dir, transform=train_transform)
print('Size of training dataset :', len(trainset))

testset = CustomDataSet(test_dir, transform=test_transform)
print('Size of test dataset :', len(testset))


trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=512, 
                                               shuffle=True, 
                                               num_workers=4)

testloader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=28,
                                              shuffle=False,
                                              num_workers=4)

print(f"class to index mapping: {trainset.class_to_idx}")
len(testloader)


modelm = models.mobilenet_v2(pretrained=True)

# Freeze parameters of the trained network 
for param in modelm.parameters():
    param.requires_grad = False
    
#print the model to check the classifer and change it
print (modelm.classifier)

# define new classifier and append it to our Mobilenet model
modelm.classifier= nn.Sequential(nn.Dropout(p=0.6, inplace=False),
                                nn.Linear(in_features=1280, out_features=29, bias=True),
                                nn.LogSoftmax(dim=1))


# unlock last three blocks before the classifier(last layer).
for p in modelm.features[-1].parameters():
    p.requires_grad = True

#loss function
criterion = nn.NLLLoss()

#optimizer to train only the classifier and the previous three block.
optimizer = torch.optim.Adam([{'params':modelm.features[-1].parameters()},
                        {'params':modelm.classifier.parameters()}], lr=0.0005)

#Learning Rate scheduler to decrease the learning rate by multiplying it by 0.1 after each epoch on the data.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

print(modelm.classifier)


# cuda details

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# how many GPU's do we have
print("The number of GPU's we have are : ", torch.cuda.device_count() )

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  modelm = nn.DataParallel(modelm)

# model to GPU's
modelm = modelm.to(device)

#Define number of epochs through data and run the training loop
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modelm.to(device)
epochs = 1
step = 0
running_loss = 0
print_every = 30
trainlossarr=[]
testlossarr=[]
oldacc=0

steps=math.ceil(len(trainset)/(trainloader.batch_size))


#TrainClass

#%%time
from tqdm import tqdm
import sys
from colorama import Fore,Style
no_of_folds=1
for i in range(no_of_folds):
    #print('Fold={}'.format(i))
    #tr_loader, val_loader,targets = train_fold(i+1)
    epochs = 6
    
    acc = {}
    loss_score = {}
    time_rec = {}

    for j in range(epochs):
        print(Style.RESET_ALL)
        print(f"--------------------------------- START OF EPOCH [ {j+1} ] >>> LR =  {optimizer.param_groups[-1]['lr']} ---------------------------------\n")
        loss_arr = []
        for inputs, labels in tqdm(trainloader,desc=Fore.GREEN + f"* progess in EPOCH {j+1} ",file=sys.stdout):
            modelm.train()
            start_time = time.time()
            step += 1
            inputs=inputs.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()

            props = modelm.forward(inputs)
            loss = criterion(props, labels)
            loss.backward()
            optimizer.step()
            loss_arr.append(loss.item())

            running_loss += loss.item()

            if (step % print_every == 0) or (step==steps):
                test_loss = 0
                accuracy = 0
                print("time for epoch", time.time() - start_time)
                time_rec[j] = time.time() - start_time
                loss_score[j] = sum(loss_arr)/len(loss_arr)
                modelm.eval()
                tqdm._instances.clear()
                with torch.no_grad():
                    for inputs, labels in tqdm(testloader,desc=Fore.BLUE + f"* CALCULATING TESTING LOSS {j+1} ",file=sys.stdout,leave=False):
                        inputs, labels = inputs.to(device), labels.to(device)
                        props = modelm.forward(inputs)
                        batch_loss = criterion(props, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(props)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        acc[j] = accuracy
 
                    
                        
                print(Style.RESET_ALL)
                tqdm.write(f"Epoch ({j+1} of {epochs}) ... "
                      f"Step  ({step:3d} of {steps}) ... "
                      f"Train loss: {running_loss/print_every:.3f} ... "
                      f"Test loss: {test_loss/len(testloader):.3f} ... "
                      f"Test accuracy: {accuracy/len(testloader):.3f} ")
                trainlossarr.append(running_loss/print_every)
                testlossarr.append(test_loss/len(testloader))
                running_loss = 0
            
        
        scheduler.step()
        step=0

        with open('plot_acc.csv', 'w') as f:
          for key in acc.keys():
            f.write("%s,%s\n"%(key,acc[key]))
    
        with open('plot_loss.csv', 'w') as f:
          for key in loss_score.keys():
            f.write("%s,%s\n"%(key,loss_score[key]))
     
    
        with open('plot_time.csv', 'w') as f:
          for key in time_rec.keys():
            f.write("%s,%s\n"%(key,time_rec[key]))


        torch.save(modelm.state_dict(), 'mobnet_gpu_4_{}.pth'.format(i))