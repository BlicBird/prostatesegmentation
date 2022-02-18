import os
import numpy as np
import torch
from torchvision import models
import h5py
import random
from torchvision import transforms as tr
from torchvision.transforms import Compose
from network import *
from segmentation import *
from dataloader import *

use_cuda = torch.cuda.is_available()


def train_VGG(epoch, train_model, optimizer, train_loader):
    total_loss = 0.0
    # correct_pred = 0.0
    total_accuracy = 0.0
    for ii, (frame, label, segmentation) in enumerate(train_loader):
        if use_cuda:
            frame, label = frame.cuda(), label.cuda()
        optimizer.zero_grad()
        preds = train_model(frame)
        loss = criterion(preds, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # if torch.all(preds.round() == label):
        # correct_pred += 1
        accuracy = (preds.round() == label).sum() / (2 * float(preds.shape[0]))
        total_accuracy += accuracy.item()

    loss_mean = total_loss / len(train_loader)
    # accuracy = correct_pred/len(train_loader)
    accuracy = total_accuracy / len(train_loader)

    return loss_mean, accuracy


def validation_VGG(epoch, model, optimizer, validation_loader):
    moving_loss = 0.0
    # correct_pred = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for ii, (frame, label, segmentation) in enumerate(validation_loader):
            if use_cuda:
                frame, label = frame.cuda(), label.cuda()
            optimizer.zero_grad()
            preds = model(frame)
            loss = criterion(preds, label)
            moving_loss += loss.item()
            # if torch.all(preds.round() == label):
            # correct_pred += 1
            accuracy = (preds.round() == label).sum() / (2 * float(preds.shape[0]))
            total_accuracy += accuracy.item()

        loss_mean = moving_loss / len(validation_loader)
        # accuracy = correct_pred/len(validation_loader)
        accuracy = total_accuracy / len(validation_loader)

    return loss_mean, accuracy


def modded(model):
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.5, inplace=False),
        torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.5, inplace=False),
        torch.nn.Linear(in_features=4096, out_features=2, bias=True),
        torch.nn.Sigmoid())
    return model


vgg16_bn = models.vgg16_bn()
model_vgg = modded(vgg16_bn)

if use_cuda:
    model_vgg.cuda()

criterion = torch.nn.BCELoss()
optimizer_vgg = torch.optim.Adam(model_vgg.parameters(), lr=1e-4)
folder_name = 'drive/My Drive/MedicalImaging/dataset70-200.h5'

data_vgg = H5Dataset_classification(folder_name)
transforms = Compose([tr.ToPILImage(), tr.RandomRotation(degrees=90), tr.RandomHorizontalFlip(p=0.5),
                      tr.RandomVerticalFlip(p=0.3), tr.ToTensor()])

train_data_vgg, validation_data_vgg, test_data_vgg = torch.utils.data.random_split(data_vgg, [100, 50, 50])
augment_data_vgg = Transform_classification(train_data_vgg, transforms)
train_data_comb_vgg = torch.utils.data.ConcatDataset([train_data_vgg, augment_data_vgg])

train_loader_vgg = torch.utils.data.DataLoader(train_data_comb_vgg, batch_size=8, shuffle=True)
validation_loader_vgg = torch.utils.data.DataLoader(validation_data_vgg, batch_size=8, shuffle=True)
test_loader_vgg = torch.utils.data.DataLoader(test_data_vgg, batch_size=1, shuffle=True)

## Training
epoch_number = 500
for epoch in range(1,epoch_number):
  print('Epoch:', epoch)
  train_loss_vgg, train_accuracy = train_VGG(epoch,model_vgg,optimizer_vgg,train_loader_vgg)
  print('Training: Loss = ', train_loss_vgg, 'Accuracy = ', train_accuracy)
  validation_loss_vgg, validation_accuracy = validation_VGG(epoch,model_vgg,optimizer_vgg,validation_loader_vgg)
  print('Validation: Loss = ', validation_loss_vgg, 'Accuracy = ', validation_accuracy)
  print('-------------------------------------------------------------------------')

#print(model_vgg.state_dict().keys())
path = "/content/drive/My Drive/MedicalImaging/classifier.pt"
torch.save(model_vgg.state_dict(), path)

path = "/content/drive/My Drive/MedicalImaging/classifier.pt"
model_vgg.load_state_dict(torch.load(path))

test_screened_frames = []
test_screened_labels = []
for (frame,label,segmentation) in test_loader_vgg:
  if use_cuda:
    frame, label, segmentation = frame.cuda(), label.cuda(), segmentation.cuda()
  pred = model_vgg(frame).cpu()
  pred = torch.round(pred)
  if pred.tolist()[0] == [1,0]:
    test_screened_frames.append(torch.squeeze(frame.cpu(),dim=0))
    test_screened_labels.append(torch.squeeze(segmentation.cpu(),dim=0))

test_data_screen = Classification_screen(test_screened_frames,test_screened_labels)
#print(len(test_data_screen))

test_data_screened = torch.utils.data.DataLoader(test_data_screen,batch_size=1,shuffle=True)
average_test_pred_list_seg1_screen, average_test_pred_list_seg2_screen, test_frames_screen, test_labels_screen = test_UNet(model_UNet_1,model_UNet_2,model_UNet_3, model_UNet_4, test_data_screened)
average_predictions_seg1_screen = ((torch.reshape(torch.stack(average_test_pred_list_seg1_screen), (2, len(test_data_screen) , 58, 52))).sum(dim=0) >= 1).long()
average_predictions_seg2_screen = ((torch.reshape(torch.stack(average_test_pred_list_seg2_screen), (2, len(test_data_screen) , 58, 52))).sum(dim=0) >= 1).long()

total_accuracy_seg1_screen = 0.0
total_loss_seg1_screen = 0.0
accuracy_per_frame_seg1_screen = []
for count, i in enumerate(average_predictions_seg1_screen):
    pred = torch.unsqueeze(torch.unsqueeze(i,dim=0),dim=0)
    total_accuracy_seg1_screen += jaccard(i,test_labels_screen[count]).item()
    total_loss_seg1_screen += loss_dice(pred,test_labels_screen[count]).item()
    accuracy_per_frame_seg1_screen.append(jaccard(i,test_labels_screen[count]).item())

accuracy_seg1_screen = total_accuracy_seg1_screen/len(average_predictions_seg1_screen)
loss_seg1_screen = total_loss_seg1_screen/len(average_predictions_seg1_screen)

print('Loss (test data) of segmentation 1 after 500 epochs :', loss_seg1_screen)
print('Accuracy (test data) of segmentation 1 after 500 epochs :', accuracy_seg1_screen)

total_accuracy_seg2_screen = 0.0
total_loss_seg2_screen = 0.0
accuracy_per_frame_seg2_screen = []
for count, i in enumerate(average_predictions_seg2_screen):
    pred = torch.unsqueeze(torch.unsqueeze(i,dim=0),dim=0)
    total_accuracy_seg2_screen += jaccard(i,test_labels_screen[count]).item()
    total_loss_seg2_screen += loss_dice(pred,test_labels_screen[count]).item()
    accuracy_per_frame_seg2_screen.append(jaccard(i,test_labels_screen[count]).item())

accuracy_seg2_screen = total_accuracy_seg2_screen/len(average_predictions_seg2_screen)
loss_seg2_screen = total_loss_seg2_screen/len(average_predictions_seg2_screen)

print('Loss (test data) of segmentation 1 after 500 epochs :', loss_seg2_screen)
print('Accuracy (test data) of segmentation 1 after 500 epochs :', accuracy_seg2_screen)

print(accuracy_per_frame_seg1_screen)
print(accuracy_per_frame_seg2_screen)

accuracy_comp_mean_screen = []
accuracy_comp_diff_screen = []
for (item1, item2) in zip(accuracy_per_frame_seg1_screen, accuracy_per_frame_seg2_screen):
    accuracy_comp_mean_screen.append((item1+item2)/2)
    accuracy_comp_diff_screen.append(item1-item2)

print(accuracy_comp_mean_screen)
print(accuracy_comp_diff_screen)

standard_deviation_screen = np.std(accuracy_comp_diff_screen)
mean_diff_screen = np.mean(accuracy_comp_diff_screen)
print(standard_deviation_screen)
print(mean_diff_screen)


plt.scatter(accuracy_comp_mean_screen,accuracy_comp_diff_screen,color='red')
plt.axhline(mean_diff_screen,color='red',label='Mean Difference')
plt.axhline(mean_diff_screen +1.96*standard_deviation_screen,color='green',label=('+1.96 \u03C3'))
plt.axhline(mean_diff_screen -1.96*standard_deviation_screen,color='yellow',label=('+1.96 \u03C3_'))
plt.xlabel('Mean accuracy (IoU loss) of segmentation labels methods')
plt.ylabel('Difference in accuracy (IoU loss)')
plt.legend(loc='upper left')
plt.savefig("/content/drive/My Drive/MedicalImaging/Bland_Altman_2.png")
    




