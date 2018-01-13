import numpy as np
from keras.preprocessing import image
from keras.utils import np_utils


class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_x = 224, dim_y =224, dim_z = 3, batch_size = 32, shuffle = True, num_classes=10):
      'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.dim_z = dim_z
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.num_classes = num_classes

  def generate(self, labels, list_IDs):
      'Generates batches of samples'
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
              labels_temp = [labels[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
              # Generate data
              X, y = self.__data_generation(labels_temp, list_IDs_temp)

              yield X, y

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def __data_generation(self, labels, list_IDs_temp):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
      y = np.empty((self.batch_size, self.num_classes))

      # Generate data
      for i, ID in enumerate(list_IDs_temp):

          img = image.load_img(ID, target_size=(self.dim_x, self.dim_y))
          data = image.img_to_array(img)
          data = data.astype('float32')
          data = data / 255
          # Apply Mean subtraction
          data -= np.mean(data)
          # Apply Normalization
          data /= np.std(data)

          # Store volume
          X[i, :, :, :] = data
          # Store class
          y[i, :] = np_utils.to_categorical(labels[i], self.num_classes)

      return X, y

