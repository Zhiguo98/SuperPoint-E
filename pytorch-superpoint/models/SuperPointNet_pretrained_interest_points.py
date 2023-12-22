"""
Network to load pretrained model from Magicleap.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from PIL import Image
# Create the preprocessing transformation here

import torchvision.transforms as transforms

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2




def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )

class SuperPointNet_pretrained(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet_pretrained, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)

    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return semi, desc





###############################
class PoseExpNet(nn.Module):

    def __init__(self, nb_ref_imgs=2, output_exp=False):
        super(PoseExpNet, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(3*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])

        self.pose_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)

        if self.output_exp:
            upconv_planes = [256, 128, 64, 32, 16]
            self.upconv5 = upconv(conv_planes[4],   upconv_planes[0])
            self.upconv4 = upconv(upconv_planes[0], upconv_planes[1])
            self.upconv3 = upconv(upconv_planes[1], upconv_planes[2])
            self.upconv2 = upconv(upconv_planes[2], upconv_planes[3])
            self.upconv1 = upconv(upconv_planes[3], upconv_planes[4])

            self.predict_mask4 = nn.Conv2d(upconv_planes[1], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask3 = nn.Conv2d(upconv_planes[2], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask2 = nn.Conv2d(upconv_planes[3], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask1 = nn.Conv2d(upconv_planes[4], self.nb_ref_imgs, kernel_size=3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.nb_ref_imgs)
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

        if self.output_exp:
            out_upconv5 = self.upconv5(out_conv5  )[:, :, 0:out_conv4.size(2), 0:out_conv4.size(3)]
            out_upconv4 = self.upconv4(out_upconv5)[:, :, 0:out_conv3.size(2), 0:out_conv3.size(3)]
            out_upconv3 = self.upconv3(out_upconv4)[:, :, 0:out_conv2.size(2), 0:out_conv2.size(3)]
            out_upconv2 = self.upconv2(out_upconv3)[:, :, 0:out_conv1.size(2), 0:out_conv1.size(3)]
            out_upconv1 = self.upconv1(out_upconv2)[:, :, 0:input.size(2), 0:input.size(3)]

            exp_mask4 = nn.functional.sigmoid(self.predict_mask4(out_upconv4))
            exp_mask3 = nn.functional.sigmoid(self.predict_mask3(out_upconv3))
            exp_mask2 = nn.functional.sigmoid(self.predict_mask2(out_upconv2))
            exp_mask1 = nn.functional.sigmoid(self.predict_mask1(out_upconv1))
        else:
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None

        if self.training:
            return [exp_mask1, exp_mask2, exp_mask3, exp_mask4], pose
        else:
            return exp_mask1, pose


def extract_and_plot_keypoints(semi_processed, original_image, threshold):
    """
    Extracts keypoints from the processed semi tensor and plots them on the original image.

    Parameters:
    semi_processed: The processed output from the semi head of the network.
    original_image: The original image on which keypoints will be plotted.
    threshold: Threshold value to consider a point as a keypoint.

    Returns:
    None
    """
    # Extract keypoints
    keypoints = np.where(semi_processed[0] > threshold)
    keypoints = np.array(keypoints).T

    # Plot keypoints on the original image
    plt.figure(figsize=(5, 5))
    plt.imshow(original_image, cmap='gray')
    plt.scatter(keypoints[:, 1], keypoints[:, 0], c='r', s=0.1)  # Red color, small size
    plt.title("Keypoints on Image")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperPointNet_pretrained()
    model = model.to(device)
    model.load_state_dict(torch.load("/Users/zhiguoma/Desktop/master/homework/image_understanding/final/SuperPointTrackingAdaptation/pytorch-superpoint/pretrained/superpoint_v1.pth"))
    
    from torchsummary import summary
    summary(model, input_size=(1, 224, 224))

    print(model.eval())

    # Processing the images and using the forward function
    # Replace with your image path
    color_image_path = "/Users/zhiguoma/Desktop/master/homework/image_understanding/dataset/EndoJPEG/hyperK_000/00030.jpg"
    color_image = Image.open(color_image_path).convert('L')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    gray_image = transform(color_image).unsqueeze(0)
    [semi, desc] = model.forward(gray_image)
    print("Semi size:", semi.shape)
    print("Desc size:", desc.shape)

    # Interest Point Decoder
    N = 1
    Hc = semi.shape[2]
    Wc = semi.shape[3]
    print(semi)
    semi_softmax = F.softmax(semi, dim=1)
    interest_points = semi_softmax[:, :-1, :, :]

    semi_processed = F.interpolate(interest_points, scale_factor=8, mode='bilinear', align_corners=False)
    print("Decoded Semi shape:", semi_processed.shape)

    # Descriptor Decoder
    D = 256
    desc_interpolated = F.interpolate(desc, scale_factor=8, mode='bicubic', align_corners=False)
    desc_normalized = F.normalize(desc_interpolated, p=2, dim=1)
    print("Decoded Desc shape:", desc_normalized.shape)

    original_image = cv2.imread(color_image_path, cv2.IMREAD_COLOR)

    interest_points_mask = semi_processed[0, 0, :, :] > 0.0001
    binary_mask = interest_points_mask.to(torch.uint8)
    interest_points_coords = np.argwhere(binary_mask.numpy())

    # Plot the interest points on the initial image
    color = (0, 255, 0)

    counter = 0
    for i in range(0, len(interest_points_coords), 64):
        point = interest_points_coords[i]
        cv2.circle(original_image, tuple(reversed(point)), 1, color, -1)

    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Interest Points Overlay")

    plt.show()
