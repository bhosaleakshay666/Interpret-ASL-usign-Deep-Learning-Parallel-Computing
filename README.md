
**1. Introduction:**

```
1.1 Background:
Speech impairment is a disability that affects a person’s ability to speak and hear. Deaf communities
have always faced difficulties in communicating with others and always need to be accompanied by a
translator. Although sign language is omnipresent, it is a challenge for these communities to
communicate with non-sign language speakers. The alternative of written communication is
cumbersome, impersonal and even impractical when an emergency occurs. In order to diminish this
obstacle and to enable dynamic communication, we present an ASL recognition system that uses
Convolutional Neural Networks (CNN) in real time to translate a video of a user’s ASL signs into text.
With advances in deep learning, there has been headway in the fields of motion and gesture
recognition.
American Sign Language (ASL) is a natural language that serves as a predominant sign language of deaf
communities in the United States. ASL comprises a set of 26 signs which are the American alphabets.
ASL involves grammatical aspect which is marked by incorporating rhythmic & circular movement while
punctual aspect is achieved by modifying the sign so that it has a stationary hand position.
```
```
1.2 Motivation:
According to the World Health Organization (WHO), the number of hearing-impaired individuals has
reached over 400 million. It is a fact that signers and impaired individuals always need a companion to
translate for them. There are many poverty-stricken countries which cannot offer the same luxury.
Businesses struggle with speech barriers thus losing valuable connections with individuals from the
deaf communities.
There is a need for point-to-point communication to maintain an efficient environment. Automated
systems and computer-generated tasks now handle most analysis and predictions, thus giving rise to
the era of Machine Learning. The motivation behind this study was to create our own machine
designed to learn human features and gestures for signed interpretations. Establishing a novel model
that can be used for predicting and classifying ASL without the need of a physical translator.
```
```
1.3 Goals:
The machines are fed with an array of labeled images or raw data. These images are trained using a
high-level machine learning algorithm that seeks to find relationships among features. Once completed
the machine will be able to apply an algorithm best suited for pattern recognition. Alphabets and
numbers can be predicted with optimal accuracy.
Various pre-trained Convolutional Neural Network (CNN) Architectures and proposed deep learning
methodologies have been applied and compared to recognize sign language and develop an efficient
learning model with increased recognition rate. Developments in deep-learning and computer vision
have already provided valuable products in vehicle detection, face recognition and audio manipulation.
The goal is to provide a novel solution to determine pattern recognition in images. The model will be
then parallelized using parallel machine learning techniques and execution time is calculated.
```

```
3 | Page
```
**2. Methodology**

```
We have come up with a methodology to use Pytorch which will help us to build and train a deep
learning model. This model would help us to classify images in the test and well as the train datasets
into 29 classes. These 29 classes constitute of the 26 ASL alphabets, space, delete and nothing. These
characters help the deaf community to communicate with people and computers.
With this dataset we use a pre-trained mobile network. We will define a classifier and connect it to this
network and then train the classifier on our image dataset.
The methodology is based on below steps:
● Obtaining the ASL images dataset from Kaggle
● perform exploratory data analysis and define transformers for processing images
● define the classifier to have an output of 29 layers
● train the classifier with the pretrained model available
● test the trained model
```
**2.1 MobileNetV2 Architecture**
As the name applied, the MobileNet model is designed to be used in mobile applications. MobileNet
uses depth wise separable convolutions. It significantly reduces the number of parameters when
compared to the network with regular convolutions with the same depth in the nets.
MobileNetV2 introduces a new CNN layer, the inverted residual and linear bottleneck layer,
enabling high accuracy/performance in mobile and embedded vision applications. The new layer
builds
on the depth-wise separable convolutions introduced in MobileNet. The MobileNetV2 network is built
around this new layer and can be adapted to perform object classification and detection.
The premise of the inverted residual layer is that a) feature maps are able to be encoded in
low-dimensional subspaces and b) non-linear activations result in information loss in spite of their
ability to increase representational complexity. These principles guide the design of the new
convolutional layer.


As inferred from the above block diagram, the pretrained MobileNetV2 model used in our implementation
follows below steps:
● The iteration consists of two block strides. The initial convolution of size 1*1 is done and an activation
function of Relu6 is applied.
● Rectified Linear Unit Function is given as: A(x) = max(0,x) , where x is the output of the hidden layer.
The ReLu function is as shown above. It gives an output x if x is positive and 0 otherwise.
● Rectified Linear Unit 6 (Relu6) is an activation function which is used in deep convolutional neural
networks and is used to further optimize the MobileNetV2 architecture.
● The range of ReLu is [0, inf). This means it can blow up the activation, which also is not a favorable
condition for a network. If you multiply a bunch of terms which are greater than 1, they will tend
towards infinity, hence exploding gradient as you get further from the output layer if you have
activation functions which have a slope > 1.
● First we cap the units at 6, so our Relu activation function is: y=min(max(x,0),6)


```
● The MobileNetV2 uses a depth-wise separable convolution layer with filter 3x3. Again a linear
convolution of 1x1 is applied.
● In the second layer of V2 where stride 2 is initialized, it follows the same steps as stride but this time
the layer does not get added again.
```
We used the pretrained mobilenetv2 model and applied it to our new implementation.

We freezed the parameters of the layers so that the pretrained model does not get trained again.

We created our custom classifier by using different parameters like Sequential, Linear and LogSoftMax. The
Sequential Neural Network Layer is a plain stack of layers that has one input tensor and one output tensor.
Linear Layer uses matrix multiplication to transform the input layer to an output layer using a weighted matrix.
Logsoftmax is the final activation layer applied on a neural network which creates an output tensor.

Loss function is used to minimize the loss to achieve optimal output.

Adam optimizer is used to decrease the rate of error while training neural networks. It is efficient for image
recognition problems. The learning rate is the most important hyperparameter so it is vital to know the effects
of the learning rate on model performance and to build an intuition about the dynamics of the learning rate
on model behavior.


**3. Description of Dataset:**

The dataset consists of two subfolders, namely the training set and the test set. Each contains images in
further subfolders depicting individual alphabets, spaces, deletions, hand gestures and a class of nothing
images for better training of the model. All the images are in JPG formats.

What we are classifying: We use the test set images and classify the hand gestures as ASL alphabets spaces
and deletions.

The entire dataset is stored in the “scratch” directory of the Discovery cluster.

```
➢ Size of the Dataset:
○ 5 GB
➢ Train Set:
○ Size: 4.64 GB
○ 223074 Images in 29 Folders
➢ Test Set:
○ 28 Images in 28 Folders
```
The test set has a folder less than the training set because in the training set a folder has been dedicated to
“Nothing”, a set of images that do not have any hand gestures in them for better training purposes.

Data Source

Link to Kaggle dataset: https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-
dataset


**4. Results & Analysis**

**Environment Details:**

**Discovery Cluster:**

Discovery is a high-performance computing (HPC) resource for the Northeastern University research
community. The Discovery cluster is located in Holyoke’s Massachusetts Green High Performance Computing
Center (MGHPCC). The computing resources are shared between five institutions: Northeastern, BU, Harvard,
MIT and UMass.

Northeastern University dedicates a reservation on the cluster for our course. The reservation is in the name
of CSYE7105-gpu.

**Commands:**

$ ssh -Y <username> @ login.discovery.neu.edu

$ srun -p gpu --gres=gpu:1 --pty /bin/bash``

$ srun -p reservation --reservation=csye7105-gpu --gres=gpu:1 --mem=100G --pty /bin/bash``

$ srun -p reservation --reservation=csye7105-gpu --gres=gpu:4 --mem=120G --pty /bin/bash``

$ module load anaconda3/2022.

$ module load cuda/11.

$ conda create --name pytorch_env python=3.7 anaconda -y

$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y

$ source activate python_env

$ conda activate pytorch_env

Since we are using PyTorch, we go ahead and establish its environment over the cluster followed by activating
the environments.

Then we install the necessary packages of PyTorch for using functions to transform the image dataset


### GPU:


**Experiments:**

As we mentioned earlier we attach a custom defined classifier to our model, therefore although pretrained we
freeze all of its layers except a last few. The freezing of its layers enables us to only train the custom layers
that we attach to it, making sure not all the network is trained during the training process. We unfreeze the
last few layers and define our custom classifier that shall give us the 29 outputs as per our desire.

With multiple GPUs available on the cluster we divide the training tasks between different GPUs. Using
PyTorch’s Cuda package (torch.cuda) we use Cuda tensors over the GPU for computation. PyTorch enables us
to access the multiple GPUs and then run our training function and later run our model.

Before implementing any of the parallel computing logic we check the device available, and in case we do
have a GPU, we further check for the number of GPUs available using above snips of scripts from PyTorch.
Thus deciding if we can implement the “nn.DataParallel” function or not.


**nn.DataParallel:** This function defined under PyTorch’s cuda package ensures parallel execution across
multiple GPUs. This function parallelizes the models training bysplitting the the inputs across available devices,
each receiving a chunk of the data, and copies of model, its parameters and weights in the forward pass. Now
after a run the gradients computed on each GPU are collected and synchronized in the Backward Pass making
the entire multi-tiered operation appear as a whole. Hence after execution of every model, the function
collects and merges the synced results before returning the result to the us. To sum it up, it replicates a model
on multiple GPU devices if available, distributes the input in the first dimension, then gathers and combines
the synchronized results in the first dimension and returns the final result to the user.

Moving ahead, we have executed our Deep learning model across different GPUs, a single GPU on our local
device, single GPU on the cluster, 2 GPUs on the cluster, followed by 3 and 4 GPUs on the cluster. For all the
below experiments we have a standard epoch count set to 6 epochs over different GPU numbers. The batch
size for the training set is set as 512 while for the testing set it is set to 28.Following are the snippets of the
execution on the cluster for each GPU number showing the first and the last epoch.


**Experiment #1:**

**With GPU’s #1 and Data Parallelism: No**


**Results:**


**Experiment #**

**With 2 GPU’s and Data Parallelism: Yes!**



**Results:**

**Experiment #3:**


**With 3 GPU’s and Data Parallelism: Yes!**

**Results:**


- Experiment #


**With 4 GPU’s and Data Parallelism: Yes!**


**Results:**



**Transforms:**

The transform function does normalization of the images as defined by the PyTorch documentation, apart
from cropping, scaling and resizing and rotating the images, all done randomly before converting it to Tensors
for easier use with PyTorch. If only we could have figured out the right transformations, we could have
improved the accuracy of the entire model. The transformations we used were simply textbooks and we
should have experimented a little bit more around it.

**Dataloader:**

To help with the complexity of loading data and preparing it for the pipeline of a neural network we use the
DataLoader functionality of PyTorch. It processes the data in batches, picking them from the datasets and
shuffling them for a randomised training approach.


**Training Function:**

The training function is invoked in the above nested for loops. The first for loop is added to make sure if in
future we need cross validation then we can add that by simply changing a value of the no_of_folds variable.
The entire second loop iterates over the number of epochs. After that the third loop invokes the timer and the
training function, it passes the input and its respective labels, which in our case are the alphabets, to the GPU
device. The inner fourth loop is where the model checks the test set and calculates the loss function and
accuracy. Before finishing the execution of this process the plot accuracies, losses and times are stored to csv
files for each epochs.


**Analysis:**

Comparing each of the above accuracy, loss and time values together **:**

**Accuracy Analysis**


### Epochs GPU 1 GPU 2 GPU 3 GPU 4

### 1 0.535 0.678 0.642 0.607

### 2 0.714 0.642 0.607 0.678

### 3 0.642 0.607 0.571 0.75

### 4 0.607 0.464 0.642 0.535

### 5 0.571 0.535 0.714 0.671

### 6 0.571 0.678 0.535 0.678

As evident in the above table, increasing and spanning the model over higher number of GPUs did not affect
the accuracy of the model starkly. Although with 3 GPUs it looks like the accuracy went down a little. But using
2 GPUs and 4 GPUs we can see that the accuracy increased as compared to execution over single GPU.
Therefore we conclude that as long as the functionality of the model with respect to input data, layers in the
network, epochs and batch sizes are handled properly, spanning and stretching the model for parallel
execution on multiple GPU yields almost 15-20% better accuracy at the most.



**Loss Analysis**

```
Epochs GPU 1 GPU 2 GPU 3 GPU 4
```
### 1 2.39 2.413 2.444 2.456

### 2 2.024 2.062 2.09 2.117

### 3 1.992 2.031 2.062 2.086

### 4 1.989 2.03 2.06 2.086

### 5 1.99 2.034 2.059 2.084

### 6 1.989 2.029 2.057 2.081

Irrespective of the number of GPU’s used, we observed that the loss calculated is stagnant after the first
epoch, although slight movement can be observed in the 3rd and the 4th decimal places, the loss calculated is
constant. Hence we conclude that increasing and scaling over more number of GPUs is only valid option if we
also increase the number of epochs because we can see that as the epoch count increases the loss starts to
decrease although slightly.



**Time Analysis**


### Epochs GPU 1 GPU 2 GPU 3 GPU 4

### 1 0.012487 0.024565 0.059435 0.058242

### 2 0.010672 0.057819 0.049089 0.072574

### 3 0.019231 0.036374 0.038967 0.086719

### 4 0.013499 0.025095 0.049242 0.047838

### 5 0.019072 0.02895 0.051356 0.050834

### 6 0.011532 0.024423 0.037321 0.047582

The time calculated above is in 6 decimal places although when cross checked with the ETA on the cluster we
deduced that these are the seconds values which need to be multiplied by 10000 and divided by 60 to match
the time format of the cluster. Secondly we observed that a single GPU was the fastest of all the other GPUs
irrespective of the epoch size. This might occur due to overhead functionality of chunking the data, making
copies of the entire model and its parameters and synchronizing the execution results together and then
presenting them after every epoch.



**6. Conclusion**

```
● Increasing the number of GPUs does not necessarily enhance the performance of the model. There is a
tradeoff between the number of GPUs and the performance we can achieve.
● Keeping the number of epochs constant does not necessarily impact the loss since we observe that the
loss function goes on decreasing as the number of epochs are increased. Although it’s a very minimal
change observed with respect to the number of GPUs.
● We observe that time increases as the number of GPUs increase as there is overhead cost involved due
to data chunking, data preprocessing and data distribution across various devices.
```

**References:**

**1. Data Parallelism : https://www.sciencedirect.com/topics/computer-science/data-
parallelism#:~:text=Data%20parallelism%20means%20that%20each,deep%20net%20structure%20and%20p
parameters.**

**2. PyTorch Documentation: https://pytorch.org/docs/stable/index.html**

**3. Joblib documentation: https://joblib.readthedocs.io/en/latest/parallel.html**

**4. Cuda : https://pytorch.org/docs/stable/cuda.html**

**5. Discovery Northeastern Documentation : https://rc-docs.northeastern.edu/en/latest/using-
discovery/workingwithgpu.html#specifying-a-gpu-type**

**6. Reference Notebook:**

**https://www.kaggle.com/code/stpeteishii/asl-american-sign-torch-linear**

**7. Reference Notebook:**

**https://www.kaggle.com/code/paultimothymooney/interpret-sign-language-with-deep-learning**

**8. MobileNet V2 Pytorch:**

**https://pytorch.org/hub/pytorch_vision_mobilenet_v2/**

**9. MobileNet V2:**

**https://arxiv.org/abs/1801.04381**

**10. Github Pytorch Model:**

**https://github.com/pytorch/vision/tree/main/torchvision/models**


