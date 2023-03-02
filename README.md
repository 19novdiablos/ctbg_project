# ctbg_project
Objectives of the study
Research applying computer vision combined with AI and using ResNet-50 technique to identify supporting medicinal herbs, treating liver disease. After training the model with high accuracy will be put into practice to help those wishing to learn about medicinal herbs for liver disease as quickly and accurately as possible.
Data includes:
• Number of leaves identification: 14 types of medicinal herbs to prevent, support and treat liver disease.
• Number of original photos of each type: 200-300 photos with color and clarity.
• Specifically, 14 types of medicinal herbs include:!
[image](https://user-images.githubusercontent.com/84515603/222318356-387dd2c1-1823-403b-8b68-307f1a5bf124.png)

#Build Model
Step 1: Read the path containing the image data directory.
      import pathlib
      data_url='/content/drive/MyDrive/data/#14Data_Train'
      data_dir = pathlib.Path(data_url)
Step 2: Divide the read data set into two parts, the dataset and the test set with the ratio of 90/10 using the Keras library with image_dataset_from_directory.Specify the size of the input image to be 224 x 224.
      dataset = tf.keras.preprocessing.image_dataset_from_directory(
          data_dir,
          labels='inferred',
          label_mode='categorical',
          class_names=None,
          color_mode='rgb',
          batch_size=32,
          image_size=(224, 224),
          shuffle=True,
          seed=123,
          validation_split=0.1,
          subset='training',
          interpolation='bi

      test_data = tf.keras.preprocessing.image_dataset_from_directory(
          data_dir,
          labels='inferred',
          label_mode='categorical',
          class_names=None,
          color_mode='rgb',
          batch_size=32,
          image_size=(224, 224),
          shuffle=True,
          seed=123,
          validation_split=0.1,
          subset='validation',
          interpolation='bilinear'
      )
Step 3: Define a data enhancement function because the amount of data to train the model is not enough to set requirements.

      data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.4),
         tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)
      ])
Use a lambda to map the dataset created from Step 2 with the data enhancement function defined. Then divide the dataset into two parts training and validation with the ratio of 80/20.
      # Apply data augmentation
      augmented_dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

      # Split the dataset into training and validation sets
      train_data = augmented_dataset.take(np.floor(0.8 * len(dataset)).astype(int))
      val_data = augmented_dataset.skip(np.floor(0.8 * len(dataset)).astype(int))
Step 4: Create a ResNet50 model with imagenet weights and start training the model.resnet_model = Sequential()
pretrained_model= tf.keras.applications.ResNet50(include_top=False,input_shape=(224,224,3),pooling='avg',classes=14,weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False
resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
resnet_model.add(Dropout(0.3))
resnet_model.add(Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
resnet_model.add(Dropout(0.3))
resnet_model.add(Dense(14, activation='softmax'))
 
resnet_model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

#Build Website
![image](https://user-images.githubusercontent.com/84515603/222319393-b91f0a24-7dee-40c8-9b53-55cd88e18125.png)

![image](https://user-images.githubusercontent.com/84515603/222319444-83c6b6e6-57a7-4b90-b249-6fedbee05ac5.png)

Product
![image](https://user-images.githubusercontent.com/84515603/222319484-db5b5389-f707-4d42-9f08-572c08d8391c.png)

Product detail
![image](https://user-images.githubusercontent.com/84515603/222319633-f74a80c9-9eb6-4d9c-b670-5112128712f0.png)

Classification leaf from camera
![image](https://user-images.githubusercontent.com/84515603/222319739-c3b2c06f-1b91-4312-b33c-4bd5163a5726.png)

Cart
![image](https://user-images.githubusercontent.com/84515603/222319797-f4a70a5f-8fd1-4da2-b2cc-f4785be28131.png)

Payment
![image](https://user-images.githubusercontent.com/84515603/222319835-60cca938-9bbe-4475-b450-f56e4cfb24b9.png)





