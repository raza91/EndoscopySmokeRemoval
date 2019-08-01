# EndoscopySmokeRemoval  
 A small but interesting research on medical endoscope
 
 Version-1
 
 Our method compared with other methods
![image](https://github.com/ColaBreadQAQ/EndoscopySmokeRemoval/blob/master/example/example.gif)
 Original: original haze image [1]   
 DC: Dark Channel[2]  
 AOD-Net: AOD-Net[3]  
 R: Training with real data  
 S: Training with synthesized data  
 S+R: Training with synthesized and real data  
 
 DarkChannel.py is the dehaze method using dark channel.  This program is obtained from internet:  
 HazeGen.py is the program for synthesizing haze images  
 The training program is modified from AOD-Net: https://github.com/TheFairBear/PyTorch-Image-Dehazing
 

 Reference:  
 [1] Real images in partial nephrectomy in da Vinci surgery. Website: hamlyn.doc.ic.ac.uk/vision/
 [2] Single Image Haze Removal Using Dark Channel Prior  
 [3] AOD-Net: An All-in-One Network for Dehazing and Beyond  

 