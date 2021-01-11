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

