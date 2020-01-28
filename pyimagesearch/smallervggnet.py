# importing the packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes): 
        #the depth of the image or the number of channels
        #classes is the number of classes in the dataset
        
        model = Sequential()
        input-shape= (height, width, depth)
        chan-dim = -1

        if K.image_data_format() == "channel-first":
            input-shape = (depth, height, width)
            chan-dim = 1
            
        # CONV=>RELU=>POOL
        model.add(
            Conv2D(32, (3,3), 
            padding="same",
            input_shape=(150, 150,3))
            )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size =(3,3)))
        model.add(Dropout(0.25))
            #Dropout randomly disconeect nodes from the current layer to the next layer
            #Helps introduce redundancy into the model
        
        #(CONV => RELU) *2
        model.add(
            Conv2D(64, (3,3), padding="same" )
            )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        #Stacking multiple CONV and RELU layers together (prior to reducing the spatial dimensions of volume) allows us to lear a richer set of features
        # (CONV => RELU) * 2 => POOL
        model.add(
            Conv2D(128, (3,3)),
            padding = "same"
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan-dim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # FC => RELU
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        #softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        #return the constructed network architecture
        return model


