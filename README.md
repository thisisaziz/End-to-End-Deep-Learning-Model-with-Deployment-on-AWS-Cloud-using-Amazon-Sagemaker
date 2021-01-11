# End-to-end-Deep-Learning-Model-Deployment-on-aws

So Here we'll build the model from scratch and then deploy it with the help of sagemaker and ec2 

### Basic step for end to end deployment
- launch a sagemaker notebook instance
- upload your training code in that instance
- start the training after selecting the image and instance for training 
- create an Endpoint for model deployment
- create an EC2 instance for flask web-app 



#### Creating sagemaker notebook instance 
- Go to sagemaker notebook instance and then click on create notebook

<img src="https://github.com/zerocool-11/End-to-end-Deep-Learning-Model-Deployment-on-aws/blob/main/images/sagemaker.png">

then select the configurations as  follow-

<img src="https://github.com/zerocool-11/End-to-end-Deep-Learning-Model-Deployment-on-aws/blob/main/images/notebook-2.png">

now open it in jupyter notebook  after that upload your code and dataset there with my model.ipynb and open my model.ipynb

<img src="https://github.com/zerocool-11/End-to-end-Deep-Learning-Model-Deployment-on-aws/blob/main/images/notebook-1.png">

first 3 cell will create a s3 bucket after that we will copy our dataset and flask-web-app.zip file in that s3 bucket 

<img src="https://github.com/zerocool-11/End-to-end-Deep-Learning-Model-Deployment-on-aws/blob/main/images/image.png">

after that we will create a instance for training

<img src="https://github.com/zerocool-11/End-to-end-Deep-Learning-Model-Deployment-on-aws/blob/main/images/train.png">

as you can see in the screenshot for this training we are using Tensorflow image it will have all the required libraries installed in it we just have to pass the python version we want(py_version) and tensorflow version and entry_point is our training code and that's it now we'll pass our training dataset in the fit function and the training will begin

after that we can deploy our model on sagemaker endpoint with just one line of code

<img src="https://github.com/zerocool-11/End-to-end-Deep-Learning-Model-Deployment-on-aws/blob/main/images/dep.png">

now we have deployed our model on sagemaker so let's create ec2 instance for web-app using flask

for ec2 instance we need minimum 2gb of ram so i used t2.small and ubuntu image 
simply launch an ec2 instance and then go in using ssh
In the ec2 instance install awscli and setup it using your access id and key after that copy the web-app.zip file from s3 
'''
aws s3 cp s3://web-dep-12/web-app.zip .
'''
unzip the web-app.zip
install the libraries from requirements.txt file 
and don't forget to change this endpoint name with yours

<img src="https://github.com/zerocool-11/End-to-end-Deep-Learning-Model-Deployment-on-aws/blob/main/images/endpoin.png">


