#**Traffic Sign Recognition**  

---
###Writeup / README

### All required files are here
Including the python notebook, write up and the tensorflow saved data.

###Data Set Summary & Exploration

####1. Basic Data Set Summary

The code for this step is contained in the second code cell of the IPython notebook. I used the built-in python len() function to calculate summary statistics of the traffic signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = 32*32
* Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fifth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data classes distribution.

![Data Visualization](./output1.png)

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. 

The code for this step is contained in the sixth code cell of the IPython notebook.

Here I used 3 different kinds of methods to preprocess the data.

1. Normaliza, for the RGB matrix the value varies from 0-255, at first, I thought all the value are in the certain range, so there is no need to do the normalization, however, later I found out that doing the normalization would speed up the training speed, and improve the final accuracy

2. Grayscale, in order to reduce the dimentions of the image, I applied the grayscale using the transfer matrix as [0.2989, 0.5870, 0.1140], by this step it turns the colored images into the gray image.

3. Shuffle data, I just used the scipy funciton to shuffle all the data.

Here is an example of a traffic sign image before and after grayscaling, sorry for the markdown is hard to resize the image to same size.

![Before Processing](./imag1.png)
![After Processing](./imag2.png)

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. 

For the training, validation and testing data, I use them directly from the first cell of code the IPython notebook provided.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is based on the Lenet structure, however, inorder to provide higher accuracy, I made some changed to each layer, I think lenet is good, however, for the traffic signs case, the traffic signs has more features en consisted of the following layers

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Scale images  | 
| Convolution 5x5   | 1x1 stride, Valid padding, outputs 28x28x16 |
| Max Pooling 		| 2x2 stride, ouputs 14x14x16 |
| RELU					|						|
| Convolution 5x5   | 1x1 stride, Valid padding, outputs 10x10x32 |
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 |
| RELU					|						|
| Flatten				| outputs 800 |
| Fully connected	| outputs 512      	|
| RELU					|						|
| Dropout				| Prob 0.5			|
| Fully connected	| outputs 256      	|
| RELU					|						|
| Dropout				| Prob 0.5			|
| Fully connected	| outputs 43      	|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the softmax_cross_entropy as the loss function, and used the AdamOptimizer with learning rate 0.001 to train the model. 

I used batch size as 128, and epochs as 20 to trained the model, because I used the GPU machine, I think more epochs may result in higher accuracy, however, from the training logs I found out that maybe 15 epochs is enough.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of around 0.94
* validation set accuracy of 0.941 
* test set accuracy of 0.924

My iterative approach:

* At first, the architecture I used if the same as the LeNet-5, but the result was not as good as my expectatio, the accurary was only 85% and the test results on the new images was not good.
* The problem I think about the previous architecture is because the LeNet's layers channel numbers are too small. Comparing the digits and traffic signs, the traffic signs has more complex features, so more channels would be better for this situation, so I increased the channel numbers of the architecture.
* Also, for the previous training, the epochs are 10, the performance was not good, so I changed it to 20, and reach the higher performance, however, due to the results, it shows only 15 epochs are enough for reach the final result.
* I think Dropout probability may also affect the final results, but due to my experiment, the 0.5 and 0.8 has no big difference.
* I think if I have more free time, I would like to try make the network architecture deeper.
* The visualization of the first layer of my CNN network:
![cnn_visualization](cnnn.png)
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](ped.png) ![alt text](right_next.png) ![alt text](road_work.png)
![alt text](bike.png) ![alt text](round.png) ![alt text](turn_left.png)

I think all the images would be easy to classify, because they are clean image,the background is very simple, compare to the training set.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 
The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Pedestrians      		| Pedestrians  	| 
| Right-of-way at the next intersection  | Right-of-way at the next intersection |
| Road work					| Road work	|
| Bicycles crossing	      		| Bicycles crossing |
| Roundabout mandatory		| Roundabout mandatory  |
| Turn left ahead		| Turn left ahead  |

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. 

Compare the old test set, the accuracy is higher with the current architecture, it shows the model is working well, and some of my assumptions about how the architecture affect the accuracy of model is right. Because, as I said above at the first stage, I use original LeNet5 architecture, with that, the accuracy on the new test images(same set as these ones) was only around 40% compare to the 85% on test set, I think by that time it was overfitting. But after I changed the new structure, it became better.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

 The result can be showed as:
![alt text](res.jpeg)

From the result image, we can see that the possbility of each new test image is all very close to 1.0, it shows the result the prediction is certain and accrurate, I think it has a good ability to classify the traffic signs.