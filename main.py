import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tqdm.notebook import tqdm
import pydicom
import numpy as np
import shutil
from PIL import Image
import scipy
import torch 
import torchvision
import torchvision.transforms as transforms
from torchvision import models , datasets
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

print("All modules have been imported")

labels = pd.read_csv("../input/png-dataset-for-rsna-mgmt-detection/png_data/png_voxel_converted_ds/train_labels.csv")


## downloading data
# TODO:
#   - Pending Kaggle activation key
# kaggle competitions download -c rsna-miccai-brain-tumor-radiogenomic-classification


main_folder_path = "../input/png-dataset-for-rsna-mgmt-detection/png_data/png_voxel_converted_ds"
main_train_folder_path = os.path.join(main_folder_path  , "train")
for subject in tqdm(os.listdir(main_train_folder_path)):
    subject_folder = os.path.join(main_train_folder_path , subject)
    for mri_type in os.listdir(subject_folder):
        mri_type_folder = os.path.join(subject_folder , mri_type)
        for mri_image in os.listdir(mri_type_folder):
            original_image_path = os.path.join(mri_type_folder , mri_image)
            mri_image = subject +"_"+ mri_type +"_"+ mri_image
            subject_num = int(subject)
            idx = np.where(labels['BraTS21ID'] == subject_num)[0][0]
            label = str(labels.loc[idx , 'MGMT_value'])
            new_image_folder_path =os.path.join("data" , label)
            new_image_path = os.path.join(new_image_folder_path , mri_image)
            shutil.copy(original_image_path , new_image_path)

print("Images with label 0 = " , len(os.listdir("data/0")) , "Images with label 1 = " , len(os.listdir("data/1")))


for folder in os.listdir("data"):
    folder_name = str(folder)
    path = "data/"+folder_name
    for file in tqdm(os.listdir(path)):
        img = Image.open(path + '/' + file)
        clrs = img.getcolors()
        if len(clrs) == 1:
            os.remove(path + '/' + file)


print("Images with label 0 = " , len(os.listdir("data/0")) , "Images with label 1 = " , len(os.listdir("data/1")))

'''
!mkdir "data/TRAIN"
!mkdir "data/TRAIN/1"
!mkdir "data/TRAIN/0"
!mkdir "data/VAL"
!mkdir "data/VAL/0"
!mkdir "data/VAL/1"
!mkdir "data/TEST"
!mkdir "data/TEST/0"
!mkdir "data/TEST/1"
'''

IMG_PATH = "./data"

#split the data into train/test/val
for CLASS in tqdm(["0" , "1"]):
    IMG_NUM = len(os.listdir(IMG_PATH +"/"+ CLASS))
    for (n, FILE_NAME) in enumerate(os.listdir(IMG_PATH +"/"+ CLASS)):
            img = IMG_PATH+ '/' +  CLASS + '/' + FILE_NAME
            if n <4000 :
                shutil.copy(img, 'data/TEST/' + str(CLASS) + '/' + FILE_NAME)
            elif n < 0.9*IMG_NUM:
                shutil.copy(img, 'data/TRAIN/'+ str(CLASS) + '/' + FILE_NAME)
            else:
                shutil.copy(img, 'data/VAL/'+ str(CLASS) + '/' + FILE_NAME)

'''
!rm -rf "data/0"
!rm -rf "data/1"
'''
'''
len(os.listdir("data/TRAIN/1")) , len(os.listdir("data/TRAIN/0")) , len(os.listdir("data/VAL/1")) , len(os.listdir("data/VAL/0")) , len(os.listdir("data/TEST/1")) , len(os.listdir("data/TEST/0"))
'''


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'TRAIN': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    
    'VAL': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    
    'TEST': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['TRAIN', 'VAL' , 'TEST']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['TRAIN', 'VAL' , 'TEST']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['TRAIN', 'VAL' , 'TEST']}
class_names = image_datasets['TRAIN'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(class_names)


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()

# Get a batch of training data
inputs, classes = next(iter(dataloaders['TRAIN']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


def train_model(model, model_name, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['TRAIN' , 'VAL']:
            if phase == 'TRAIN':
                model.train()  # Set model to training mode
            elif phase == 'VAL':
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'TRAIN'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'TRAIN':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'TRAIN':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'VAL' and epoch_acc > best_acc:
                best_acc = epoch_acc
                #best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model , model_name+'weights.pt')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


'''
Transfer Learning
    ResNet
    - Resnet was introduced in the paper Deep Residual Learning for Image Recognition. 
    There are several variants of different sizes, including Resnet18, Resnet34, Resnet50, Resnet101, and Resnet152, 
    all of which are available from torchvision models. Here we use Resnet18 just for demonstration purpose. 
    You can change the model by just changing "resnet18" to "resnetxyz" , xyz being the number.
'''

resnet = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features

resnet.fc = nn.Linear(num_ftrs, 2)
resnet = resnet.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(resnet.parameters(), lr=0.001)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_model(resnet,"resnet", criterion, optimizer, step_lr_scheduler, num_epochs=1)


'''
    AlexNet
    - Alexnet was introduced in the paper ImageNet Classification with Deep Convolutional Neural Networks and was 
    the first very successful CNN on the ImageNet dataset. When we print the model architecture, we see the model 
    output comes from the 6th layer of the classifier.
'''

alexnet = models.alexnet(pretrained=True)
num_ftrs = alexnet.classifier[6].in_features

alexnet.classifier[6] = nn.Linear(num_ftrs,2)
alexnet = alexnet.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(alexnet.parameters(), lr=0.001)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_model(alexnet ,"alexnet", criterion, optimizer, step_lr_scheduler, num_epochs=1)


'''
    VGG
    VGG was introduced in the paper Very Deep Convolutional Networks for Large-Scale Image Recognition. 
    Torchvision offers eight versions of VGG with various lengths and some that have batch normalizations layers. 
    Here we use VGG-11 with batch normalization. The output layer is similar to Alexnet.
'''

vgg = models.vgg11_bn(pretrained=True)
num_ftrs = vgg.classifier[6].in_features

vgg.classifier[6] = nn.Linear(num_ftrs,2)
vgg = vgg.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(vgg.parameters(), lr=0.001)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_model(vgg,"vgg", criterion, optimizer, step_lr_scheduler, num_epochs=1)