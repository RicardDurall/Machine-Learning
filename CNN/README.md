In this folder a CNN is used to try to solve a classification problem.

For more details about the problem check the <a href="https://www.kaggle.com/c/plant-seedlings-classification">Kaggle </a> competiton.

Usage:

config.yml -> it contains the main parameters of the CNN.

Predict.py -> it makes the prediction and produces a .csv as an output (for the competition).

CNN.py -> it has implemented the training and it is used a generator provided by Keras.

CNN_generator -> it has implemented the training and it is used a handmade generator.

DataGenerator -> handmade generator.

Note:
<a href="https://www.kaggle.com/c/plant-seedlings-classification/data">Kaggle </a> offers a train archive. This has been splited into train and validation to feed the CNN.

