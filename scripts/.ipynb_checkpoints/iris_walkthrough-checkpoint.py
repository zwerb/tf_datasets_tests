#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras


import time


DEBUG = True

LOCAL_EXECUTION = False #if you want to run this off-Internet

    ## Update Required: need to make this go recursively up until it finds the project dir 
PROJECT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    
    ## Update Required: need to make this go recursively up until it finds the project dir 
DATASET_PATH = os.path.abspath(os.path.join(PROJECT_PATH,'datasets')) 
MODELS_PATH = os.path.abspath(os.path.join(PROJECT_PATH,'models')) 

TRAINING_DATASET_DIRNAME = 'train'
TEST_DATASET_DIRNAME = 'test'
TRAINING_DATASET_PATH = os.path.join(DATASET_PATH,TRAINING_DATASET_DIRNAME)
TEST_DATASET_PATH = os.path.join(DATASET_PATH,TEST_DATASET_DIRNAME)

DATASET_PATH_SAFE = DATASET_PATH.replace(" ", "\ ")
TRAINING_DATASET_PATH_SAFE = TRAINING_DATASET_PATH.replace(" ", "\ ")


if DEBUG:
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    print("PROJECT_PATH: {}".format(PROJECT_PATH))
    print("DATASET_PATH: {}".format(DATASET_PATH))
    print("MODELS_PATH: {}".format(MODELS_PATH))
    #print("DATASET_PATH_SAFE: {}".format(DATASET_PATH_SAFE))
    print("TRAINING_DATASET_PATH: {}".format(TRAINING_DATASET_PATH))
    #print("TRAINING_DATASET_PATH_SAFE: {}".format(TRAINING_DATASET_PATH))
    print("TEST_DATASET_PATH: {}".format(TEST_DATASET_PATH))
    #print("TEST_DATASET_PATH_SAFE: {}".format(TEST_DATASET_PATH))
    


train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fn = os.path.basename(train_dataset_url)
train_dataset_fp = tf.keras.utils.get_file(fname=train_dataset_fn,
                                           origin=train_dataset_url,
                                           cache_dir=DATASET_PATH,
                                           cache_subdir=TRAINING_DATASET_DIRNAME)

train_dataset_fp_safe = str(train_dataset_fp).replace(" ","\ ")
print("Local copy of the dataset file: {}".format(train_dataset_fp))
print("Safe path: {}".format(train_dataset_fp_safe))


get_ipython().system('head -n5 {train_dataset_fp_safe}  ## the first row can be the column names, or info-data that is ignored')


## column order in CSV file - explicitly setting the column and feature names
## the first row of the data will be ignored either way
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
select_columns = [0,1,2,3,4]

feature_names = column_names[:-1] #features are all columns up to last
label_name = column_names[-1] #last column is the "label"

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))


## class names
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']


batch_size = 32

    # returns a tf.data.Dataset of ({feature},label) pairs)
    # constructor picks up dataset names if they are not explicitly given below
    # column_names, if not given, it will take them from head row
    # header=True skips the first row
    ## Update Required: how to explicitly set the label column in params
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    header=True,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)


features, labels = next(iter(train_dataset))

print(features)
print(labels)


plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()


plt.scatter(features['petal_width'],
            features['sepal_width'],
            c=labels,
            cmap='viridis')

plt.xlabel("petal_width")
plt.ylabel("sepal_width")
plt.show()


def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


train_dataset = train_dataset.map(pack_features_vector)


features, labels = next(iter(train_dataset))

print(features[:1])
print(features[:5])
print(features[:11])


model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])


predictions = model(features)
predictions[:5]


tf.nn.softmax(predictions[:5])


print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_)


l = loss(model, features, labels, training=False)
print("Loss test: {}".format(l))


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels, training=True).numpy()))


## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    epoch_accuracy.update_state(y, model(x, training=True))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))


fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()


test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
test_dataset_fn = os.path.basename(test_url)

test_fp = tf.keras.utils.get_file(fname=test_dataset_fn,
                                    origin=test_url,
                                    cache_dir=DATASET_PATH,
                                    cache_subdir=TEST_DATASET_DIRNAME)

test_fp_safe = str(test_fp).replace(" ","\ ")

print("Local copy of the test dataset file: {}".format(test_fp))
print("Safe test path: {}".format(test_fp_safe))


test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)


test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = model(x, training=False)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


tf.stack([y,prediction],axis=1)


predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])

# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
predictions = model(predict_dataset, training=False)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))


model_name = "test_model-" + time.strftime("%Y-%m-%d-%H%M%S")

model.save(os.path.join(MODELS_PATH,model_name))


model_name = "test_model-" + "2020-09-26-193045"

new_model = keras.models.load_model(os.path.join(MODELS_PATH,model_name))


for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = new_model(x, training=False)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))




