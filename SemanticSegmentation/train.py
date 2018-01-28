import Segnet
import os
import yaml
from keras import __version__ as keras_version
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger


import keras
from datetime import datetime  # for filename conventions
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from tensorflow.python.lib.io import file_io  # for better file I/O
import sys

from DataGenerator import DataGenerator

# import config file
with open("trainer/config1.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

train_dir = cfg['train_dir']
train_label = cfg['train_label']
val_dir = cfg['val_dir']
val_label = cfg['val_label']
test_dir = cfg['test_dir']
weights_dir = cfg['weights_dir']
log_dir = cfg['log_dir']

model_selected = cfg['model_selected']
input_height = cfg['input_height']
input_width = cfg['input_width']
num_classes = cfg['num_classes']
epochs = cfg['epochs']
optimizer = cfg['optimizer']
batch_size = cfg['batch_size']

# Parameters
params = {'dim_x': input_height,
          'dim_y': input_width,
          'dim_z': 3,
          'batch_size': batch_size,
          'shuffle': True,
          'num_classes': num_classes}


# Get directories
def get_images(data_dir, label_dir):
    test_data_file=[]
    test_data_path=[]
    for subdir, dirs, files in os.walk(data_dir):
        try:
            for file in files:
                test_data_file.append(os.path.join(subdir, file))
        except:
            print("This class doesnt exist {}".format(subdir))
	for subdir, dirs, files in os.walk(label_dir):
		try:
			for file in files:
				test_data_path.append(os.path.join(subdir, file))
		except:
			print("This class doesnt exist {}".format(subdir))


    return test_data_file, test_data_path
# Select model and optimizator
def get_model():

	if model_selected == 'segnet':
		model, model_h, model_w = Segnet.segnet(num_classes, img_h=input_height, img_w=input_width)


	model.summary()
	# Create optimizer
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	return model, model_h, model_w

# Running model
def run_models():


	# train_data, train_target, train_id = read_and_normalize_train_data()
	X_train, Y_train = get_images(train_dir,train_label)
	X_valid, Y_valid = get_images(val_dir,val_label)

	# callbacks functions
	checkpoint = ModelCheckpoint(weights_dir, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
	earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')
	csv_logger = CSVLogger(log_dir)
	callbacks = [checkpoint, earlystop, csv_logger]


	model, model_h, model_w  = get_model()

	# Generators
	training_generator = DataGenerator(**params).generate(Y_train, X_train, model_h, model_w )
	validation_generator = DataGenerator(**params).generate(Y_valid, X_valid, model_h, model_w )


	# Train model on dataset
	model.fit_generator(generator=training_generator,
						steps_per_epoch=len(X_train) // batch_size,
						epochs=epochs,
						validation_data=validation_generator,
						validation_steps=len(X_valid) // batch_size,
						verbose=1),
						callbacks=callbacks)



if __name__ == '__main__':
	print('Keras version: {}'.format(keras_version))
	run_models()



