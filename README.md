# Face-Recognition-Using-Fine-tuning-with-VGG16
In this project, I have implemented facial recognition using a deep learning model. I have used the VGG16 pre-trained model and apply fine-tuning network surgery. Basically what happened is after the image is captured from the camera and the face is extracted from it we resize the resolution of cropped faces by 224x224 because the vgg16 model takes input as a 224x224 image.  After that captured resized images are augmentative to increase the size of a dataset because data is like food for AI.  

And then the real thing happens Fine tunning  
 Remove the fully connected nodes at the end of the network (i.e., where the actual class label predictions are made). 
 Replace the fully connected nodes with freshly initialized ones. 
 Freeze earlier CONV layers earlier in the network (ensuring that any previous features learned by the CNN are not destroyed). 
 Start training, but only train the FC layer heads. 
 Optionally unfreeze some/all of the CONV layers in the network and perform the second pass of training.
