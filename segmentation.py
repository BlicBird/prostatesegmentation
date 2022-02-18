import torch
from network import *
from dataloader import *
from torchvision import transforms as tr
from torchvision.transforms import Compose

use_cuda = torch.cuda.is_available()


def loss_dice(y_pred, y_true, eps=1e-6):
    numerator = torch.sum(y_true * y_pred, dim=(2, 3)) * 2
    denominator = torch.sum(y_true, dim=(2, 3)) + torch.sum(y_pred, dim=(2, 3)) + eps
    return torch.mean(1. - (numerator / denominator))


def jaccard(y_pred, y_true):
    intersection = torch.sum(y_true * y_pred, dim=(2, 3))
    total = torch.sum(y_pred + y_true, dim=(2, 3))
    union = total - intersection
    return torch.mean((intersection / union))


def train_UNet(epoch, model, optimizer, loader):
    total_loss = 0.0
    total_IOU = 0.0
    for ii, (frame, label) in enumerate(loader):
        if use_cuda:
            frame, label = frame.cuda(), label.cuda()
        optimizer.zero_grad()
        preds = model(frame)
        loss = loss_dice(preds, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        IOU = jaccard(preds, label)
        total_IOU += IOU.item()

    loss_mean = total_loss / len(loader)
    IOU_mean = total_IOU / len(loader)

    return loss_mean, IOU_mean


def validation_UNet(epoch, model, optimizer, validation_loader):
    total_loss = 0.0
    total_IOU = 0.0
    with torch.no_grad():
        for ii, (frame, label) in enumerate(validation_loader):
            if use_cuda:
                frame, label = frame.cuda(), label.cuda()
            optimizer.zero_grad()
            preds = model(frame)
            loss = loss_dice(preds, label)
            total_loss += loss.item()
            IOU = jaccard(preds, label)
            total_IOU += IOU.item()

        loss_mean = total_loss / len(validation_loader)
        IOU_mean = total_IOU / len(validation_loader)
    return loss_mean, IOU_mean


def test_UNet(model_1, model_2, model_3, model_4, test_loader):
    models_test_predictions_seg1 = []
    models_test_predictions_seg2 = []
    model_frames = []
    model_labels = []

    with torch.no_grad():
        for ii, (frame, label) in enumerate(test_loader):
            if use_cuda:
                frame, label = frame.cuda(), label.cuda()
                model_frames.append(frame)
                model_labels.append(label)
                model_1_pred = model_1(frame)
                model_2_pred = model_2(frame)
                model_3_pred = model_3(frame)
                model_4_pred = model_4(frame)
                models_test_predictions_seg1.append(model_1_pred)
                models_test_predictions_seg1.append(model_2_pred)
                models_test_predictions_seg2.append(model_3_pred)
                models_test_predictions_seg2.append(model_4_pred)

    return models_test_predictions_seg1, models_test_predictions_seg2, model_frames, model_labels


### Segmentation 1 ###
model_UNet_1 = UNet()
if use_cuda:
    model_UNet_1.cuda()

model_UNet_2 = UNet()
if use_cuda:
    model_UNet_2.cuda()

optimizer_1 = torch.optim.Adam(model_UNet_1.parameters(), lr=1e-4)
optimizer_2 = torch.optim.Adam(model_UNet_2.parameters(), lr=1e-4)
folder_name = 'drive/My Drive/MedicalImaging/dataset70-200.h5'

data = H5Dataset(folder_name)
transforms = Compose([tr.ToPILImage(), tr.RandomRotation(degrees=90), tr.RandomHorizontalFlip(p=0.5),
                      tr.RandomVerticalFlip(p=0.3), tr.ToTensor()])
torch.manual_seed(20)
data_1, data_2, test_data = torch.utils.data.random_split(data, [80, 80, 40])

# Sample 1 data
train_data_1, validation_data_1 = torch.utils.data.random_split(data_1, [60, 20])
augment_data_1 = Transform(train_data_1, transforms)
train_data_comb_1 = torch.utils.data.ConcatDataset([train_data_1, augment_data_1])

train_loader_1 = torch.utils.data.DataLoader(train_data_comb_1, batch_size=8, shuffle=True)
validation_loader_1 = torch.utils.data.DataLoader(validation_data_1, batch_size=8, shuffle=True)

# Sample 2 data
train_data_2, validation_data_2 = torch.utils.data.random_split(data_2, [60, 20])
augment_data_2 = Transform(train_data_2, transforms)
train_data_comb_2 = torch.utils.data.ConcatDataset([train_data_2, augment_data_2])

train_loader_2 = torch.utils.data.DataLoader(train_data_comb_2, batch_size=8, shuffle=True)
validation_loader_2 = torch.utils.data.DataLoader(validation_data_2, batch_size=8, shuffle=True)

# test_loader = torch.utils.data.DataLoader(test_data,batch_size=1,shuffle=True)


### Segmentation2 ###
model_UNet_3 = UNet()
if use_cuda:
    model_UNet_3.cuda()

model_UNet_4 = UNet()
if use_cuda:
    model_UNet_4.cuda()

optimizer_3 = torch.optim.Adam(model_UNet_3.parameters(), lr=1e-4)
optimizer_4 = torch.optim.Adam(model_UNet_4.parameters(), lr=1e-4)
folder_name = 'drive/My Drive/MedicalImaging/dataset70-200.h5'

data_seg2 = H5Dataset2(folder_name)
torch.manual_seed(20)
data_seg2_1, data_seg2_2, test_data_seg2 = torch.utils.data.random_split(data_seg2, [80, 80, 40])

## Sample 1
train_data_seg2_1, validation_data_seg2_1 = torch.utils.data.random_split(data_seg2_1, [60, 20])
augment_data_seg2_1 = Transform(train_data_seg2_1, transforms)
train_data_comb_seg2_1 = torch.utils.data.ConcatDataset([train_data_seg2_1, augment_data_seg2_1])

train_loader_seg2_1 = torch.utils.data.DataLoader(train_data_comb_seg2_1, batch_size=8, shuffle=True)
validation_loader_seg2_1 = torch.utils.data.DataLoader(validation_data_seg2_1, batch_size=8, shuffle=True)

##Sample 2
train_data_seg2_2, validation_data_seg2_2 = torch.utils.data.random_split(data_seg2_2, [60, 20])
augment_data_seg2_2 = Transform(train_data_seg2_2, transforms)
train_data_comb_seg2_2 = torch.utils.data.ConcatDataset([train_data_seg2_2, augment_data_seg2_2])

train_loader_seg2_2 = torch.utils.data.DataLoader(train_data_comb_seg2_2, batch_size=8, shuffle=True)
validation_loader_seg2_2 = torch.utils.data.DataLoader(validation_data_seg2_2, batch_size=8, shuffle=True)

# test_loader_seg2 = torch.utils.data.DataLoader(test_data_seg2,batch_size=1,shuffle=True)

#### Training begin ####

### Training Segmentation 1 ###

epoch_num = 501

for epoch in range(1,epoch_num):
  print('Model 1: Epoch:',epoch)
  train_loss,train_IOU = train_UNet(epoch,model_UNet_1,optimizer_1,train_loader_1)
  print('Training: Loss=',train_loss,'IOU=',train_IOU)
  validation_loss, validation_IOU = validation_UNet(epoch,model_UNet_1,optimizer_1,validation_loader_1)
  print('Validation: Loss=',validation_loss,'IOU=',validation_IOU)
  print('------------------------------------------------------------------------')

path1 = "/content/drive/My Drive/MedicalImaging/UNet1.pt"
torch.save(model_UNet_1.state_dict(), path1)


for epoch in range(1,epoch_num):
  print('Model 2: Epoch:',epoch)
  train_loss,train_IOU = train_UNet(epoch,model_UNet_2,optimizer_2,train_loader_2)
  print('Training: Loss=',train_loss,'IOU=',train_IOU)
  validation_loss, validation_IOU = validation_UNet(epoch,model_UNet_2,optimizer_2,validation_loader_2)
  print('Validation: Loss=',validation_loss,'IOU=',validation_IOU)
  print('------------------------------------------------------------------------')

path2 = "/content/drive/My Drive/MedicalImaging/UNet2.pt"
torch.save(model_UNet_2.state_dict(), path2)

### Training Segmentation 2 ###

epoch_num = 501

for epoch in range(1,epoch_num):
    print('Model 3: Epoch:',epoch)
    train_loss,train_IOU = train_UNet(epoch,model_UNet_3,optimizer_3,train_loader_seg2_1)
    print('Training: Loss=',train_loss,'IOU=',train_IOU)
    validation_loss, validation_IOU = validation_UNet(epoch,model_UNet_3,optimizer_3,validation_loader_seg2_1)
    print('Validation: Loss=',validation_loss,'IOU=',validation_IOU)
    print('------------------------------------------------------------------------')

path3 = "/content/drive/My Drive/MedicalImaging/UNet3.pt"
torch.save(model_UNet_3.state_dict(), path3)

for epoch in range(1,epoch_num):
    print('Model 4: Epoch:',epoch)
    train_loss,train_IOU = train_UNet(epoch,model_UNet_4,optimizer_4,train_loader_seg2_2)
    print('Training: Loss=',train_loss,'IOU=',train_IOU)
    validation_loss, validation_IOU = validation_UNet(epoch,model_UNet_4,optimizer_4,validation_loader_seg2_2)
    print('Validation: Loss=',validation_loss,'IOU=',validation_IOU)
    print('------------------------------------------------------------------------')

path4 = "/content/drive/My Drive/MedicalImaging/UNet4.pt"
torch.save(model_UNet_4.state_dict(), path4)

#### Training done ####

import matplotlib.pyplot as plt

#Segmentation 1
path1 = "/content/drive/My Drive/MedicalImaging/UNet1.pt"
model_UNet_1.load_state_dict(torch.load(path1))

path2 = "/content/drive/My Drive/MedicalImaging/UNet2.pt"
model_UNet_2.load_state_dict(torch.load(path2))

#Segmentation 2
path3 = "/content/drive/My Drive/MedicalImaging/UNet3.pt"
model_UNet_3.load_state_dict(torch.load(path3))

path4 = "/content/drive/My Drive/MedicalImaging/UNet4.pt"
model_UNet_4.load_state_dict(torch.load(path4))

data_comparison = H5Dataset2(folder_name, random=True)
torch.manual_seed(20)
train_data_comp,validation_data_comp,test_data_comp = torch.utils.data.random_split(data_comparison,[80,80,40])
test_loader_comp = torch.utils.data.DataLoader(test_data_comp,batch_size=1,shuffle=False)

average_test_pred_list_seg1, average_test_pred_list_seg2, test_frames, test_labels = test_UNet(model_UNet_1,model_UNet_2,model_UNet_3, model_UNet_4, test_loader_comp)
average_predictions_seg1 = ((torch.reshape(torch.stack(average_test_pred_list_seg1), (2, len(test_data_comp) , 58, 52))).sum(dim=0) >= 1).long()
average_predictions_seg2 = ((torch.reshape(torch.stack(average_test_pred_list_seg2), (2, len(test_data_comp) , 58, 52))).sum(dim=0) >= 1).long()

total_accuracy_seg1 = 0.0
total_loss_seg1 = 0.0
accuracy_per_frame_seg1 = []
for count, i in enumerate(average_predictions_seg1):
    pred = torch.unsqueeze(torch.unsqueeze(i,dim=0),dim=0)
    total_accuracy_seg1 += jaccard(i,test_labels[count]).item()
    total_loss_seg1 += loss_dice(pred,test_labels[count]).item()
    accuracy_per_frame_seg1.append(jaccard(i,test_labels[count]).item())

accuracy_seg1 = total_accuracy_seg1/len(average_predictions_seg1)
loss_seg1 = total_loss_seg1/len(average_predictions_seg1)

print('Loss (test data) of segmentation 1 after 500 epochs :', loss_seg1)
print('Accuracy (test data) of segmentation 1 after 500 epochs :', accuracy_seg1)

total_accuracy_seg2 = 0.0
total_loss_seg2 = 0.0
accuracy_per_frame_seg2 = []
for count, i in enumerate(average_predictions_seg2):
  pred = torch.unsqueeze(torch.unsqueeze(i,dim=0),dim=0)
  total_accuracy_seg2 += jaccard(i,test_labels[count]).item()
  total_loss_seg2 += loss_dice(pred,test_labels[count]).item()
  accuracy_per_frame_seg2.append(jaccard(i,test_labels[count]).item())

accuracy_seg2 = total_accuracy_seg2/len(average_predictions_seg2)
loss_seg2 = total_loss_seg2/len(average_predictions_seg2)

print('Loss (test data) of segmentation 2 after 500 epochs :', loss_seg2)
print('Accuracy (test data) of segmentation 2 after 500 epochs :', accuracy_seg2)

print(accuracy_per_frame_seg1)
print(accuracy_per_frame_seg2)

accuracy_comp_mean = []
accuracy_comp_diff = []
for (item1, item2) in zip(accuracy_per_frame_seg1, accuracy_per_frame_seg2):
    accuracy_comp_mean.append((item1+item2)/2)
    accuracy_comp_diff.append(item1-item2)

print(accuracy_comp_mean)
print(accuracy_comp_diff)

standard_deviation = np.std(accuracy_comp_diff)
mean_diff = np.mean(accuracy_comp_diff)
print(standard_deviation)
print(mean_diff)


plt.scatter(accuracy_comp_mean,accuracy_comp_diff,color='red')
plt.axhline(mean_diff,color='red',label='Mean Difference')
plt.axhline(mean_diff +1.96*standard_deviation,color='green',label=('+1.96 \u03C3'))
plt.axhline(mean_diff -1.96*standard_deviation,color='yellow',label=('-1.96 \u03C3'))
plt.xlabel('Mean accuracy (IoU loss) of segmentation labels methods')
plt.ylabel('Difference in accuracy (IoU loss)')
plt.legend(loc='upper left')
plt.savefig("/content/drive/My Drive/MedicalImaging/Bland_Altman_2.png")

### Constructing overlay images ###

plt.imshow(average_predictions_seg1[9].cpu())
plt.savefig("/content/drive/My Drive/MedicalImaging/pred_seg1.png")

plt.imshow(average_predictions_seg2[9].cpu())
plt.savefig("/content/drive/My Drive/MedicalImaging/pred_seg2.png")

plt.imshow(torch.squeeze(torch.squeeze(test_labels[9].cpu(),dim=0),dim=0))
plt.savefig("/content/drive/My Drive/MedicalImaging/label.png")

plt.imshow(torch.squeeze(torch.squeeze(test_frames[9].cpu(),dim=0),dim=0))
plt.savefig("/content/drive/My Drive/MedicalImaging/frame.png")

from PIL import Image

pred_image_seg1 = Image.open("/content/drive/My Drive/MedicalImaging/pred_seg1.png")
pred_image_seg2 = Image.open("/content/drive/My Drive/MedicalImaging/pred_seg2.png")
frame_image = Image.open("/content/drive/My Drive/MedicalImaging/frame.png")
label_image = Image.open("/content/drive/My Drive/MedicalImaging/label.png")


blended_pred_seg1 = Image.blend(frame_image, pred_image_seg1, alpha=.3)
blended_pred_seg2 = Image.blend(frame_image, pred_image_seg2, alpha=.3)
blended_label = Image.blend(frame_image, label_image, alpha=.3)

print('After 500 epochs:')
print('Ground Truth:')
display(blended_label)
blended_label.save("/content/drive/My Drive/MedicalImaging/groundtruth_overlay_2.png")
print('Segmentation 1:')
display(blended_pred_seg1)
blended_pred_seg1.save("/content/drive/My Drive/MedicalImaging/seg1_overlay_2.png")
print('Segmentation 2:')
display(blended_pred_seg2)
blended_pred_seg2.save("/content/drive/My Drive/MedicalImaging/seg2_overlay_2.png")
