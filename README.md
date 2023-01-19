# ResNet-50
## Implementation of ResNet-50 Network
The following is an implemenation of 'Deep Residual Learning for Image Recognition' by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. It allows you to train a model with identical specifications to ResNet-50 on training and validation images saved in a specified directory.

## Basic Architecture
Our network consists of a simple convolutional layer, followed by a series of residual blocks. The residual blocks come in two varieties: identity blocks and convolutional blocks. Identity blocks add their input directly their output before activation, and are used when input and output dimensions are the same. Convolutional blocks feed their input through a projection shortcut, or convolutional layer with 1x1 kernels, to reshape the input to the appropriate output size. Downsampling is performed by using a stride length of 2. Finally, we have a fully connected layer with 1000 units and softmax activaiton.

## Model Construction Details
### Optimization
ResNet utilizes stochastic gradient descent with momentum and constant weight decay, with values of 0.9 and 0.0001 respectivley. The learning rate is initialized to 0.1 and is decreased by a factor of 10 when validation error stops decreasing. The exact criteria for this shrinking are not specified by the authors, so this implementation uses a fairly standard heuristic.

### Batch Normalization
Due to the depth of residual networks, it comes as no surprise that batch normalization is used extensively. They are used after every convolutional layer and before final activation in each residual block.

### Data Augmentations
ResNet employs several types of image augmentations. The first form of data augmentation is to rescale all images to a random value from 256 to 480 along the short side. The next step is to both train and predict on 224x224 slices of our images. Next we reflect the image horizontally with probability 0.5. RGB values are first altered by centering around zero and scaling by 255. Lastly, PCA analysis is performed over the entirety of our RGB values and the PCA terms are used to randomly jitter our original RGB values.

While the set of data augmentations is identical during both training and prediction, the method that they are applied is slightly different during these phases. Firstly, upon initilization, our model calculates the PCA terms needed for augmentation. Next, during training, each scaled input image is randomly flipped horizontally with probability 0.5, and then a random 224x224 slice of the image is chosen. The RGB values are then transformed, and image is sent into our network. Finally, upon prediction, each input image is scaled to five separate scales. Each is used to create 10 new images on which to predict: the corners and center of our original image create 5 of our training images, and their horizontal reflections create the remaining 5. The PCA values of these 10 images are then transformed, and predictions are made across these 50 new images. Our final prediction is then the average across these 50 predictions.

### Additional Implementation Details
The paper includes many finer details that are less novel. Relu activation is used for non-output layers, weight initializations are specified, etc. Each of these details are accounted for, but are likely of little interest.

## Example Implementation
### Basic Implemenation: Creating and Fitting a ResNet-50 Model
Let's create and train a ResNet-50 inspired model. Before we begin, we must ensure our data is in the desired format. We assume that all images are located in a single directory with subdirectories 'train/', 'val/', and 'test/'. The test directory need not be present. Additionally, we assume that we have a json file mapping image ids to their corresponding label, and another json file mapping labels to label index. The label indices can be arbitrary, but must be provided for the sake of consistency across sessions. The variable 'data_path' may be either a local path or a URL link to the desired image directory, and both 'label_path' and 'label_encoding' must be json files.

Let's begin by defining these paths.


```python
from ResNet import ResNet

data_path = 'DIRECTORY-CONTAINING-IMAGE-SUBDIRECTORIES'

# Example file format: {image_1: 'cat', image_2: 'mug', ...}
label_path = 'PATH-TO-IMAGE-LABEL-MAP.json'

# Example file format: {'cat': 0, 'mug': 1, ...}
label_encoding = 'PATH-TO-LABEL-INDEX-MAP.json'

```

With the paths prepared, creating and fitting our model is as simple as running the following:


```python
# Create and fit model. Store history for plotting training curve.
model = ResNet(data_path, label_path, label_encoding)
history = model.fit(epochs=5)

```

And just like that, we're ready to make predictions with our model. The 'predict' method can be used to return the 1,000 dimensional output array, or the method 'predict_n' can be used to return the n classes with the highest probability. All image augmentations are handled automatically, so the location of our color input image is the only input that is needed.



```python
test_image = '/Users/justinsima/dir/implementations/datasets/ImageNet/dummy_data/test/ILSVRC2012_test_00018560.JPEG'

pred = model.predict(test_image)
top_pred = model.predict_n(test_image, n=1)

print(f'Prediction Probabilities: \n{pred}')
print(f'Most Likely Class: {top_pred})
```

## Source
Thanks for checking out my repo. For more information please see the original paper here:
https://arxiv.org/abs/1512.03385
