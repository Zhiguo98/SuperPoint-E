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


def prepare_descriptors(desc):
    desc = desc.squeeze(0)  # 去除批次维度
    desc = desc.permute(1, 2, 0)  # 调整维度为 HxWxD
    desc = desc.reshape(-1, desc.shape[2])  # 转换为 Nx256
    return desc.detach().cpu().numpy()  # 转换为NumPy数组

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperPointNet_pretrained()
    model = model.to(device)
    model.load_state_dict(torch.load("/Users/zhiguoma/Desktop/master/homework/image_understanding/final/SuperPointTrackingAdaptation/pytorch-superpoint/pretrained/superpoint_v1.pth"))
    # model.load_state_dict(torch.load(
    #     "/Users/zhiguoma/Desktop/master/homework/image_understanding/superPointNet_400000_checkpoint.pth.tar",
    #     map_location=device)['model_state_dict'])

    # check keras-like model summary using torchsummary
    from torchsummary import summary
    summary(model, input_size=(1, 224, 224))

    print(model.eval())

    color_image_path1 = "/Users/zhiguoma/Desktop/master/homework/image_understanding/dataset/EndoJPEG/hyperK_000/00177.jpg"
    color_image_path2 = "/Users/zhiguoma/Desktop/master/homework/image_understanding/dataset/EndoJPEG/hyperK_000/00178.jpg"
    color_image1 = Image.open(color_image_path1).convert('L')
    color_image2 = Image.open(color_image_path2).convert('L')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    gray_image1 = transform(color_image1).unsqueeze(0)
    [semi1, desc1] = model.forward(gray_image1)
    gray_image2 = transform(color_image2).unsqueeze(0)
    [semi2, desc2] = model.forward(gray_image2)
    # print(semi.shape)
    # print(desc.shape)

    # Interest Point Decoder
    N = 1
    Hc = semi1.shape[2]
    Wc = semi1.shape[3]
    print(semi1)
    semi_softmax1 = F.softmax(semi1, dim=1)
    interest_points1 = semi_softmax1[:, :-1, :, :]

    semi_softmax2 = F.softmax(semi2, dim=1)
    interest_points2 = semi_softmax2[:, :-1, :, :]

    semi_processed1 = F.interpolate(interest_points1, scale_factor=8, mode='bilinear', align_corners=False)
    semi_processed2 = F.interpolate(interest_points2, scale_factor=8, mode='bilinear', align_corners=False)
    print("Decoded points shape:", semi_processed1.shape)

    # Descriptor Decoder
    D = 256
    desc_interpolated1 = F.interpolate(desc1, scale_factor=8, mode='bicubic', align_corners=False)
    desc_normalized1 = F.normalize(desc_interpolated1, p=2, dim=1)

    desc_interpolated2 = F.interpolate(desc2, scale_factor=8, mode='bicubic', align_corners=False)
    desc_normalized2 = F.normalize(desc_interpolated2, p=2, dim=1)
    # print("Decoded descriptor shape:", desc_normalized1.shape)
    desc_normalized1 = prepare_descriptors(desc_normalized1)
    desc_normalized2 = prepare_descriptors(desc_normalized2)
    print("Decoded descriptor shape:", desc_normalized1.shape)

    interest_points_mask1 = semi_processed1[0, 0, :, :] > 0.001
    binary_mask1 = interest_points_mask1.to(torch.uint8)
    interest_points_coords1 = np.argwhere(binary_mask1.numpy())
    keypoints1 = []
    keypoints2 = []
    for coord in interest_points_coords1:
        # 将坐标转换为浮点数
        x, y = float(coord[1]), float(coord[0])
        keypoints1.append(cv2.KeyPoint(x, y, 1))  # x, y, diameter

    interest_points_mask2 = semi_processed2[0, 0, :, :] > 0.001
    binary_mask2 = interest_points_mask2.to(torch.uint8)
    interest_points_coords2 = np.argwhere(binary_mask2.numpy())
    for coord in interest_points_coords2:
        # 将坐标转换为浮点数
        x, y = float(coord[1]), float(coord[0])
        keypoints2.append(cv2.KeyPoint(x, y, 1))  # x, y, diameter

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(desc_normalized1, desc_normalized2)

    img1 = cv2.imread(color_image_path1)
    img2 = cv2.imread(color_image_path2)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:9000], None, flags=2)

    plt.imshow(matched_img)
    plt.title("Matched Points")
    plt.show()


