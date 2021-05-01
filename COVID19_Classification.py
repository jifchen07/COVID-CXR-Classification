import os
import shutil
import random
import torch
import torchvision
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

# torch.manual_seed(0)

root_dir = 'database'
train_dir = 'training'
test_dir = 'testing'
classes = ['Normal', 'COVID']
BATCH_SIZE = 50
NEW_PIC_SIZE = (224, 224)
MEANS = [0.485, 0.457, 0.408]
STDS = [0.229, 0.224, 0.225]
NUM_EPOCHS = 2


'''define transformation
Normalize RGB channels by subtracting 123.68, 116.779, 103.939 and dividing by 58.393, 57.12, 57.375, respectively'''
def create_transform(new_pic_size, means, stds):
    transform_tr = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(size=new_pic_size),
         torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean=means, std=stds)]
    )

    transform_te = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(size=new_pic_size),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean=means, std=stds)]
    )

    return transform_tr, transform_te

class CXRDataset(torch.utils.data.Dataset):
    def __init__(self, class_dir: dict, transform):
        '''
        :param class_dir: a dict maps from name of the class to the folder path of its images
        :param transform:
        '''
        self.class_dir = class_dir                      # dict
        self.classes = list(self.class_dir.keys())      # list of all classes
        self.class_images = {}                          # dict maps from class name to list of images path
        self.transform = transform
        self.len = 0

        for c in self.classes:
            img_paths = [_ for _ in os.listdir(class_dir[c]) if _[-3:].lower().endswith('png')]
            self.class_images[c] = img_paths
            self.len += len(img_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        c_idx = random.randint(0, len(self.classes) - 1)
        c = self.classes[c_idx]
        idx = idx % len(self.class_images[c])
        img_path = os.path.join(self.class_dir[c], self.class_images[c][idx])
        img = Image.open(img_path).convert('RGB')       # convert for Resnet input
        return self.transform(img), c_idx

def validate(model, dl_va, loss_fn):
    model.eval()
    loss_va = 0
    accuracy = 0

    with torch.no_grad():
        for step_va, (X_va, y_va) in enumerate(dl_va):
            y_pred = model(X_va)
            loss = loss_fn(y_pred, y_va)
            loss_va += loss.item()

            _, pred = torch.max(y_pred, 1)
            accuracy += accuracy_score(y_va, pred)

    return loss_va / len(dl_va), accuracy / len(dl_va)

def train(model, dl_tr, optimizer, loss_fn):
    model.train()
    loss_tr = 0
    for step_tr, (X_tr, y_tr) in enumerate(dl_tr):
        # print(f'step: {step_tr + 1}')
        optimizer.zero_grad()
        y_pred = model(X_tr)
        loss = loss_fn(y_pred, y_tr)
        loss.backward()
        optimizer.step()
        loss_tr += loss.item()
        if step_tr % 50 == 0:
            print(f'Evaluating at step {step_tr + 1}')
            loss_va, accuracy_va = validate(resnet, dl_te, loss_fn)
            print(f'\tValidation loss: {loss_va} | Validation accuracy: {accuracy_va}')
            model.train()

    return loss_tr / len(dl_tr)


'''create the dataset'''
class_dir_tr = {}
class_dir_te = {}
for c in classes:
    class_dir_tr[c] = os.path.join(root_dir, train_dir, c)
    class_dir_te[c] = os.path.join(root_dir, test_dir, c)

transform_tr, transform_te = create_transform(NEW_PIC_SIZE, MEANS, STDS)
dataset_tr = CXRDataset(class_dir_tr, transform_tr)
dataset_te = CXRDataset(class_dir_te, transform_te)

dl_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=BATCH_SIZE, shuffle=True)
dl_te = torch.utils.data.DataLoader(dataset_te, batch_size=BATCH_SIZE, shuffle=True)


'''create model'''
resnet = torchvision.models.resnet18(pretrained=True)
resnet.fc = torch.nn.Linear(in_features=512, out_features=2)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-5)


'''train model'''
for e in range(NUM_EPOCHS):
    print(f'Training epoch {e + 1}/{NUM_EPOCHS}'.center(80, '*'))
    loss_tr = train(resnet, dl_tr, optimizer, loss_fn)
    # loss_va, accuracy_va = validate(resnet, dl_te, loss_fn)

    print(f'End of Epoch. Training loss: {loss_tr}')
    # print(f'\tValidation loss: {loss_va} | Validation accuracy: {accuracy_va}')


print('finished')
