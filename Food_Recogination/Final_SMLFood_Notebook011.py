#!/usr/bin/env python
# coding: utf-8

# ![image-2.png](attachment:image-2.png)
# 
#                                                          

#   ##                                                                                   Food Recognition
#   
#   #### Team Members:
#   #### Riya Aggarwal, PRN- 21070126070
#   #### Rohan Saraswat, PRN- 21070126071
#   #### Saksham Jain , PRN- 21070126075

# ### Problem Statement-
# Developing an image recognition system for accurately detecting and recognizing food names from images, and providing a recipe based on the recognized food item. This would enable users to easily identify the food they are interested in, and receive recipe recommendations based on their preferences.

# In[1]:


import os
import matplotlib.pyplot as plt
import pandas as np
import numpy as np
import tensorflow as tf
import cv2 as cv


# ### DATA PROCESSING AND REGULARISATION

# In[2]:


# prevents oom errors by setting gpu consumption growth
# As it uses all the potential we have to prevent it by limiting value
gpus= tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


# In[3]:


#This library is used to check the file type of images uploaded by users to ensure they are valid image files and prevent malicious uploads.
import imghdr  


# In[4]:


# Getting data from the images folder:
data_dir= "C:\\Users\\Lenovo\\Downloads\\data1" 
data="C:\\Users\\Lenovo\\Downloads\\data1"


# ### Applying Machine Learning Model:

# In[5]:


import numpy as np
import pandas as pd
import os
import cv2 as cv
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


datadir = "C:\\Users\\Lenovo\\Downloads\\data1"


# In[7]:


categories = os.listdir(datadir)


# In[8]:


# Extracting Features and labels:
# Converting in gray scale and resizing img
features = []
labels = []

for category in categories:
    category_path = os.path.join(datadir, category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (64, 64))
        features.append(img.flatten())
        labels.append(category)


# In[9]:


# Importing neccessary Library:
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[10]:


# Spliting the data:
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# In[11]:


# Applying Standard Scaler For Standardization:
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[12]:


# Training on RandomForestClassifier:
rf = RandomForestClassifier(n_estimators=100, random_state=50)
rf.fit(X_train, y_train)


# In[13]:


# predicting the Values:
y_pred = rf.predict(X_test)


# In[14]:


# Finding out the Accuracy Score:
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100)


# In[15]:


# Importing SupportVectorMachine
from sklearn.svm import SVC 


# In[16]:


# Spliting Data:
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# In[17]:


# Applying SVM:
classifier = SVC(kernel = 'rbf', random_state = 10)
classifier.fit(X_train, y_train)


# In[18]:


# Predictions:
y_pred2 = classifier.predict(X_test)


# In[19]:


# Evluating the Accuracy Score:
acc = accuracy_score(y_test,y_pred2)*100
print("Accuracy(SVM): ", acc)


# In[102]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[103]:


print(classification_report(y_test, y_pred2))


# In[ ]:





# Based on the accuracy results obtained using the RandomForestClassifier and SVM, it appears that these classifiers are not performing very well on your image dataset. The accuracy levels of 57% and 48% suggest that there may be a lot of misclassifications happening.
# 
# In this case, it might be worth considering the use of a CNN (Convolutional Neural Network) instead of these classifiers. CNNs are specifically designed for image classification tasks and have shown to perform very well in such scenarios. By using a CNN, we can leverage its ability to learn hierarchical features from the images, which can help improve the accuracy of our model.

# In[ ]:





# In[ ]:





# ## Applying  Deep Learning Model:

# Removing Unwanted Images:

# In[20]:


image_exts= ['jpeg', 'jpg','bmp','png']


# In[21]:


os.listdir(data_dir) # 11 classifications in the folder.


# In[22]:


# Printing the Image Dataset:
os.listdir(os.path.join(data_dir,'poha'))


# In[ ]:





# In[24]:


# We would be looping through the folder and would be removing unwanted images
remove_img = []
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path= os.path.join(data_dir, image_class, image)
        try:
            img= cv.imread(image_path)
            tip= imghdr.what(image_path)
            if tip not in image_exts:
                remove_img.append(image_path)
                
        except  Exception as e:
            print("Issue with image {}".format(image_path))
remove_img


# In[25]:


# iterating by the images,if unwanted image found, removing it from our dataset:
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path= os.path.join(data_dir, image_class, image)
        if(image_path in remove_img):
            os.remove(image_path)


# In[ ]:





# In[27]:


# Loading data
data= tf.keras.utils.image_dataset_from_directory("C:\\Users\\Lenovo\\Downloads\\data1")


# In[28]:


# Accessing Data PipeLine
# Converting into numpy array
data_iterator= data.as_numpy_iterator()


# In[29]:


data_iterator


# In[30]:


# For getting another batch from hte iterator batch- images and labels
batch = data_iterator.next()


# In[31]:


# Getting the length Images and Labels
len(batch)


# In[32]:


# IMages represented as numpy arrays
batch[0].shape # Batch size of 32


# In[33]:


#class_labels = ['Briyani', 'Dhokla', 'Dosa', 'Gulab_Jamun', 'Idli', 'Palak_Panner', 'ButterPaneerMasala', 'Poha', 'Vada', 'Samosa','VadaPav']
# Images : 0 and labels : 1

batch[1] # Labels: for getting new batch


# In[34]:


# Now getting the labels:
# It would help us to classify the food into catories

fig, ax= plt.subplots(ncols= 5, figsize= (20,20))
for idx, img in enumerate(batch[0][:5]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# In[ ]:





# In[36]:


# Maximum value of a given sequence:
batch[0].max()


# In[37]:


# Preprocessing Data
# Scale data for getting values btw 0 and 1 for faster model
scaled= batch[0]/ 255
scaled.max()


# In[38]:


# It speeds our data to be accessed  # Pipe line
# map is used to return dataset containing transformed elements
data= data.map(lambda x,y:(x/255,y))


# In[39]:


scaled_iterator= data.as_numpy_iterator()
scaled_iterator.next()


# In[40]:


scaled_iterator.next()[0].max()


# In[41]:


batch= scaled_iterator.next()


# In[42]:


batch[0].max()


# In[43]:


fig, ax= plt.subplots(ncols= 5, figsize= (20,20))
for idx, img in enumerate(batch[0][:5]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])
    
#Scaled Labeled Images


# ### Spliting Dataset

# In[44]:


# Spliting Dataset
len(data)
# Each 48 batch has 32 images


# In[ ]:





# In[46]:


# Assigning values
train_size= int(len(data) *.7) # 5 batches 70%
val_size= int(len(data)*.2)+1 # 2 batches
test_size= int(len(data)*.1)+1 #1 batch


# In[47]:


print(train_size, val_size, test_size)


# In[48]:


test_size+ train_size+ val_size


# In[49]:


# Applying take and skip
train= data.take(train_size)
val= data.skip(train_size).take(val_size) # skiping data we already takesn in train
test= data.skip(train_size+ val_size).take(test_size) # skipiping data tht we had taken in both


# In[50]:


val


# In[51]:


# Importing the Neccessary Imports:
from tensorflow.keras.models import Sequential
# For multiclasses:
from tensorflow.keras.utils import to_categorical

# Importing Layers, Max Polling- condensing layer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


# In[52]:


# Building neural network model for classification:
# it is built up of different layers, we would be adding layers:
model= Sequential()


# In[53]:


# 16- layers relu- gives a graph function
# 16 filters to classify
# (3,3)pixels- size, 1 stride
#Stride- no of pixcels filtered:
# Explanation:
# *The first line adds a 2D convolutional layer with 16 filters, a kernel size of (3,3), a stride of 1, and a ReLU activation function. The input shape is (256,256,3) which represents an image with a height and width of 256 pixels and 3 color channels (RGB).
# *The second line adds a max-pooling layer, which reduces the spatial dimensions of the output from the previous layer.
# *The third and fourth lines repeat the first two steps, but with 32 filters and another 16 filters respectively.
# *The fifth line adds a flatten layer that transforms the output of the previous layer into a 1-dimensional array.
# *The sixth and seventh lines add two fully connected (Dense) layers with 256 and 11 neurons, respectively. The first one uses a ReLU activation function and the second one uses a softmax activation function. The softmax activation function produces output probabilities for each of the 11 classes that the model is trained to recognize.




model.add(Conv2D(16, (3,3),1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D()) # take max value and scan and condense info

model.add(Conv2D(32, (3,3),1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3),1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten()) # condensing rows and width into 1d array in 1 value

model.add(Dense(256, activation='relu')) # 256 neurons
model.add(Dense(11, activation='softmax')) 

# Using activation functions relu and sigmoid to transfrom output layer
# Sigmoid converts value btw 0 and 1
          


# In[ ]:





# In[55]:


#Adam- Gradient based optimizer combination of Adagrad nd relu: adapts the learning rate for each parameter during training.
#LossFunction:CategoricalCrossentropy  is a standard loss function used for multi-class classification problems.
model.compile('adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()] )


# In[56]:


model.summary()


# In[ ]:





# ### Training Dataset

# In[58]:


# Train
logdir= "C:\\Users\\Lenovo\\Downloads\\Logdr"


# In[59]:


logdir


# In[60]:


tensorboard_callback= tf.keras.callbacks.TensorBoard(log_dir= logdir)


# In[61]:


train


# In[62]:


val


# In[63]:


# Converting the labels (y) to one-hot encoded vectors.
#One-hot encoding is a technique used to convert categorical data,

import tensorflow as tf

from tensorflow.keras.utils import to_categorical
num_classes = 11

train = train.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)))
val = val.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)))


# In[ ]:





# In[ ]:





# In[65]:


#The training progress and performance metrics are logged using a TensorBoard callback.
hist= model.fit(train, epochs= 10, validation_data=val,callbacks=[tensorboard_callback])


# In[ ]:





# In[66]:


# Ploting Performance: Loss
fig= plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label= 'val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()


# In[67]:


# Accuracy
fig= plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label= 'val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()


# In[68]:


# Deploying DEEP Learning Model  Shape- 256,256
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[69]:


# Saving model
tf.keras.models.save_model(model,'my_model3.hdf5')


# In[70]:


# %%writefile app.py
# import streamlit as st
# import tensorflow as tf

# st.set_option('deprecation.showfileUploderEncoding', False)
# @st.cache(allow_output_mutation=True)
# def load_model():
#     model= tf.keras.models.load_model("C:\\Users\\Lenovo\\Untitled Folder 4\\my_model2.hdf5")
#     return model


# model= load_model()
# st.write("""
#          # FOOD CLASSIFICATION  

# """)



# file= st.file_uploder("Please upload an food image", type=['jpeg', 'jpg','bmp','png'])


# import cv2
# from PIL import Image, ImageOps
# import numpy as np
# def import_and_predict(image_data,model):
#     size= (256,256)
#     image= ImageOps.fit(image,data,size,Image.ANTIALIAS)
#     img= np.asarray(image)
#     img_reshape= img[np.newaxis,...]
#     prediction= model.predict(img_reshape)
    
#     return prediction


# if file is None:
#     st.text("Please upload an Image file")
    
# else:
#     image= Image.open(file)
#     st.image(image, use_column_width=True)
#     predictions= import_and_predict(image,model)
#     class_names= ['Briyani', 'Dhokla', 'Dosa', 'Gulab_Jamun', 'Idli', 'Palak_Panner', 'ButterPaneerMasala', 'Poha', 'Vada', 'Samosa','VadaPav']
#     string= "This image most likely is: " +class_names[np.argmax(predictions)]
#     st.success(string)


# In[ ]:





# In[ ]:





# In[71]:


# Evaluation Step:
# For classification:
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# In[72]:


pre= Precision()
re= Recall()
acc= BinaryAccuracy()


# In[73]:


len(test)


# In[74]:


#evaluating the trained model on the test dataset using batch processing.
for batch in test.as_numpy_iterator():
    X,y= batch
    yhat= model.predict(X)
    y_onehot= tf.one_hot(y, depth= num_classes)
    pre.update_state(y_onehot, yhat)
    re.update_state(y_onehot,yhat)
    acc.update_state(y_onehot, yhat)


# In[101]:


# for batch in test.as_numpy_iterator():


# In[76]:


# Printing results
print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy: {acc.result().numpy()}')


# ### Evaluating:

# In[96]:


# Testing: Taking Random Image:
imgtest = cv.imread("C:\\Users\\Lenovo\\Downloads\\Food_detect_folder\\dosa_test.jpeg")
plt.imshow(cv.cvtColor(imgtest, cv.COLOR_BGR2RGB)) # For fixing color
im = cv.cvtColor(imgtest, cv.COLOR_BGR2RGB)


# In[97]:


# Resizing image
resize= tf.image.resize(im, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[ ]:





# In[79]:


# resize = resize / 255.0  # normalize the image
# resize = tf.image.grayscale_to_rgb(resize)
# resize = np.expand_dims(resize, axis=0)  # expand dimensions to include batch size dimension
# resize = tf.convert_to_tensor(resize, dtype=tf.float32)


# In[ ]:





# In[98]:


np.expand_dims(resize,0).shape


# In[81]:


# 0-Briyani,  1- Dhokla ,2- Dosa ,3- Gulab_Jamun ,4- Idli, 5- Palak_Panner, 6- ButterPaneerM, 7- Poha, 8- Vada , 9- Samosa
#10- Vada Pav


# In[99]:


# Predicting the array classes.
yhat= model.predict(np.expand_dims(resize/255,0))
yhat


# In[ ]:





# In[100]:


# Printing the predicted food type.

class_labels = ['Briyani', 'Dhokla', 'Dosa', 'Gulab_Jamun', 'Idli', 'Palak_Panner', 'ButterPaneerMasala', 'Poha', 'Vada', 'Samosa','VadaPav']

# Get the index of the highest predicted value
pred_index = np.argmax(yhat)

# Get the name of the corresponding class label
pred_class = class_labels[pred_index]

# Print the name of the predicted class and its recipe
if pred_class == 'Briyani':
    print('Briyani')
elif pred_class == 'Dhokla':
    print('Dhokla recipe')
elif pred_class == 'Dosa':
    print('Dosa recipe')
elif pred_class == 'Gulab_Jamun':
    print(' Its Gulab Jamun ')
elif pred_class == 'Idli':
    print('Idli recipe')
elif pred_class == 'Palak_Panner':
    print('Palak Panner recipe')
elif pred_class == 'Poha':
    print('Poha recipe')
elif pred_class == 'Vada':
    print('Vada recipe')
elif pred_class == 'Samosa':
    print('Samosa recipe')
elif pred_class == 'Vada Pav':
    print('Vada Pav recipe')


# In[ ]:




