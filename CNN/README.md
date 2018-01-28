<h1>CNN for classification</h1>

In this folder a CNN is used to try to solve a classification problem. Several architectures are implemented.

For more details about the problem check the <a href="https://www.kaggle.com/c/plant-seedlings-classification">Kaggle </a> competiton.

Usage:
 <ul>
  <li>config.yml -> it contains the main parameters of the CNN.</li>

  <li>Predict.py -> it makes the prediction and produces a .csv as an output (for the competition).</li>

  <li>CNN.py -> it has implemented the training and it is used a generator provided by Keras.</li>

  <li>CNN_generator -> it has implemented the training and it is used a handmade generator.</li>

  <li>DataGenerator -> handmade generator.</li>
 </ul>
<ins>Note:</ins>
<a href="https://www.kaggle.com/c/plant-seedlings-classification/data">Kaggle </a> offers a train archive. This has been splited into train and validation to feed the CNN.

