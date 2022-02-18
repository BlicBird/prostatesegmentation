import torch
from network import *
from dataloader import *
from torchvision import transforms as tr
from torchvision.transforms import Compose
from segmentation import *

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