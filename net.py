import torch
import torch.nn as nn
import math

class MyNet(nn.Module):

	def __init__(self):
		super(Unet_simple, self).__init__()
		
		self.relu = nn.ReLU(inplace=True)
		
		self.conv1_1 = nn.Conv2d(3,64,(3,3),1,1,1,1,bias=True) #in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=True
		self.conv1_2 = nn.Conv2d(64,64,(3,3),1,1,1,1,bias=True)		
		self.conv2_1 = nn.Conv2d(64,128,(3,3),1,1,1,1,bias=True)
		self.conv2_2 = nn.Conv2d(128,128,(3,3),1,1,1,1,bias=True)	
		self.conv3_1 = nn.Conv2d(128, 256, (3,3),1,1,1,1,bias=True)
		self.conv3_2 = nn.Conv2d(256, 256, (3,3),1,1,1,1,bias=True)
		self.conv4_1 = nn.Conv2d(256,512, (3,3),1,1,1,1, bias=True)
		self.conv4_2 = nn.Conv2d(512,512, (3,3),1,1,1,1, bias=True)
		
		self.maxpooling = nn.MaxPool2d((2,2),2,0,1,return_indices=False,ceil_mode=False) #kernel_size, stride, padding, dilation, return_indices=False, ceil_mode=False
		
		self.bn1 = nn.BatchNorm2d(64) #num_features, eps=1e-05, momentum=0.1, affine=True
		self.bn2 = nn.BatchNorm2d(128)
		self.bn3 = nn.BatchNorm2d(256)
		self.bn4 = nn.BatchNorm2d(512)
		self.bn5 = nn.BatchNorm2d(512) #num_features, eps=1e-05, momentum=0.1, affine=True
		self.bn6 = nn.BatchNorm2d(256)
		self.bn7 = nn.BatchNorm2d(128)
		self.bn8 = nn.BatchNorm2d(32)
		
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

		self.conv5_0 = nn.Conv2d(512,512,(3,3),1,1,1,1,bias=True)
		self.conv6_0 = nn.Conv2d(512,256,(3,3),1,1,1,1,bias=True)
		self.conv6_1 = nn.Conv2d(512,256,(3,3),1,1,1,1,bias=True)
		self.conv6_2 = nn.Conv2d(256,256,(3,3),1,1,1,1,bias=True)
		self.conv7_0 = nn.Conv2d(256,128,(3,3),1,1,1,1,bias=True)
		self.conv7_1 = nn.Conv2d(256, 128, (3,3),1,1,1,1,bias=True)
		self.conv7_2 = nn.Conv2d(128, 128, (3,3),1,1,1,1,bias=True)
		self.conv8_0 = nn.Conv2d(128,64,(3,3),1,1,1,1,bias=True)
		self.conv8_1 = nn.Conv2d(128,64, (3,3),1,1,1,1, bias=True)
		self.conv8_2 = nn.Conv2d(64,32, (3,3),1,1,1,1, bias=True)
		self.conv8_3 = nn.Conv2d(32,16, (3,3),1,1,1,1, bias=True)
		self.conv8_4 = nn.Conv2d(16,3, (1,1),1,0,1,1, bias=True)
		
		
	def forward(self, x):
		# encoder
		x1_1 = self.relu(self.conv1_1(x))
		x1_2 = self.relu(self.bn1(self.conv1_2(x1_1)))
		x2 = self.maxpooling(x1_2)
		x2_1 = self.relu(self.conv2_1(x2))
		x2_2 = self.relu(self.bn2(self.conv2_2(x2_1)))
		x3 = self.maxpooling(x2_2)
		x3_1 = self.relu(self.conv3_1(x3))
		x3_2 = self.relu(self.bn3(self.conv3_2(x3_1)))
		x4 = self.maxpooling(x3_2)
		x4_1 = self.relu(self.conv4_1(x4))
		x4_2 = self.relu(self.bn4(self.conv4_2(x4_1)))
		
		# decoder
		x5_0 = self.relu(self.bn5(self.conv5_0(x4_2)))
		x6 = self.upsample(x5_0)
		x6_0 = self.relu(self.conv6_0(x6))
		concat6 = torch.cat((x6_0, x3_2),1)
		x6_1 = self.relu(self.conv6_1(concat6))
		x6_2 = self.relu(self.bn6(self.conv6_2(x6_1)))
		x7 = self.upsample(x6_2)
		x7_0 = self.relu(self.conv7_0(x7))
		concat7 = torch.cat((x7_0, x2_2),1)
		x7_1 = self.relu(self.conv7_1(concat7))
		x7_2 = self.relu(self.bn7(self.conv7_2(x7_1)))
		x8 = self.upsample(x7_2)
		x8_0 = self.relu(self.conv8_0(x8))
		concat8 = torch.cat((x8_0, x1_2),1)
		x8_1 = self.relu(self.conv8_1(concat8))
		x8_2 = self.relu(self.bn8(self.conv8_2(x8_1)))
		x8_3 = self.relu(self.conv8_3(x8_2))
		output = self.conv8_4(x8_3)
		
		return output
