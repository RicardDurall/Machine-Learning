import numpy as np
import yaml, os
from DataGenerator import DataGenerator

#ml libraries
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.optimizers import SGD
from keras.layers import Dense, Conv2D, Activation, Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.preprocessing import image
from keras import __version__ as keras_version


CATEGORIES = ['Small-flowered Cranesbill','Cleavers', 'Scentless Mayweed' ,'Sugar beet', 'Common wheat', 'Black-grass',
              'Maize', 'Charlock', 'Loose Silky-bent', 'Common Chickweed','Shepherds Purse', 'Fat Hen']

# import config file
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


train_dir = cfg['train_dir']
val_dir = cfg['val_dir']
test_dir = cfg['test_dir']
weights_dir = cfg['weights_dir']
log_dir = cfg['log_dir']


model_selected = cfg['model_selected']
frozen_layer = cfg['model_type'][model_selected]['frozen']
size = cfg['model_type'][model_selected]['size image']

batch_size = cfg['batch_size']
num_classes = cfg['num_classes']
epochs = cfg['epochs']
data_augmentation = cfg['data_augmentation']
num_folds = cfg['num_folds']


# Parameters
params = {'dim_x': size,
          'dim_y': size,
          'dim_z': 3,
          'batch_size': batch_size,
          'shuffle': True,
          'num_classes': num_classes}


input_label=[]
input_data=[]

# Get dirs from images for training
def get_images(dir):
    input_label = []
    input_data = []
    class_num = -1
    for subdir, dirs, files in os.walk(dir):
        try:
            for file in files:
                input_data.append(os.path.join(subdir, file))
                input_label.append(class_num)
            class_num += 1
        except:
            print("This class doesnt exist {}".format(subdir))

    # Shuffle the list
    shuffle_input = list(zip(input_data, input_label))
    np.random.shuffle(shuffle_input)
    shuffle_data, shuffle_label = zip(*shuffle_input)

    return shuffle_data, shuffle_label

# Get directories
def get_dir():
    test_data_file=[]
    test_data_path=[]
    for subdir, dirs, files in os.walk(test_dir):
        try:
            for file in files:
                test_data_file.append(file)
                test_data_path.append(os.path.join(subdir, file))
        except:
            print("This class doesnt exist {}".format(subdir))


    return test_data_file, test_data_path

# Models
def create_model():
    if model_selected == 'vgg16':
        model = VGG16(weights='imagenet')
        # remove last layer
        model.layers.pop()
        # recover the output from the last layer in the model and use as input to new Dense layer
        last = model.layers[-1].output
        x = Dense(num_classes, activation="softmax")(last)
        model = Model(model.input, x)

    elif model_selected == 'mobileNet':
        model = MobileNet(alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet')
        # remove last layers
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        last = model.layers[-1].output
        x = Conv2D(num_classes, (1, 1), padding='same', name='conv_preds')(last)
        x = Activation('softmax', name='act_softmax')(x)
        x = Reshape((num_classes,), name='reshape_2')(x)
        model = Model(model.input, x)

    elif model_selected == 'xception':
        model = Xception(weights='imagenet')
        # remove last layer
        model.layers.pop()
        # recover the output from the last layer in the model and use as input to new Dense layer
        last = model.layers[-1].output
        x = Dense(num_classes, activation="softmax", name='predictions')(last)
        model = Model(model.input, x)

    return model

# Select model and optimizator
def get_model():
    model = create_model()

    # all layers
    for idx, layer in enumerate(model.layers):
        # freeze the weights
        if idx < frozen_layer:
            layer.trainable = False

    #weights = model.layers[73].get_weights()
    print(model_selected)
    model.summary()

    #Create optimizer
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True )
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def prediction(weights):

    assembale_predictios =[]
    test_data_file, test_data_path = get_dir()
    model = get_model()

    for x in xrange(len(weights)):
        predictions = []

        # load if there is an own pretrained model
        try:
            model.load_weights(weights[x])
            print('Weights loaded')
        except:
            print('Weights NOT loaded')

        for img_path in test_data_path:
            img = image.load_img(img_path, target_size=(size, size))
            x = image.img_to_array(img)
            x = x.astype('float32')
            x = x / 255
            # Apply Mean subtraction
            x -= np.mean(x)
            # Apply Normalization
            x /= np.std(x)
            x = np.expand_dims(x, axis=0)
            res_prediction = model.predict(x)
            res_prediction = res_prediction.flatten()
            predictions.append(res_prediction)

        assembale_predictios.append(predictions)

    mean_assembale = [sum(x) / len(assembale_predictios) for x in zip(*assembale_predictios)]

    res_assembale = []
    for x in mean_assembale:
        res_assembale.append(x.argmax(axis=0))

    res_assembale = [CATEGORIES[c] for c in res_assembale]

    rows = zip(test_data_file, res_assembale)

    return rows


# Running model
def run_models(num_folds=3):

    for x in xrange(num_folds):
        # train_data, train_target, train_id = read_and_normalize_train_data()
        X_train, Y_train = get_images(train_dir)
        X_valid, Y_valid = get_images(val_dir)

        # callbacks functions
        checkpoint = ModelCheckpoint(weights_dir + str(x) + ".hdf5", monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='min', period=1)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')
        csv_logger = CSVLogger(log_dir + str(x) + ".csv")
        callbacks = [checkpoint, earlystop, csv_logger]

        # Generators
        training_generator = DataGenerator(**params).generate(Y_train, X_train)
        validation_generator = DataGenerator(**params).generate(Y_valid, X_valid)

        model = get_model()

        #score = model.evaluate_generator(generator=validation_generator, steps=len(X_valid) // batch_size)

        # Train model on dataset
        model.fit_generator(generator=training_generator,
                            steps_per_epoch=len(X_train) // batch_size,
                            validation_data=validation_generator,
                            validation_steps=len(X_valid) // batch_size,
                            verbose=1,
                            epochs=epochs,
                            callbacks=callbacks)


if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    #np.random.seed(2016)
    run_models(num_folds)
