import tensorflow as tf
import tensorflow_hub as hub

import numpy as np

import time


CPU, GPU = "/device:CPU:0", "/device:GPU:0"
device = GPU

data_root = tf.keras.utils.get_file(
  'flower_photos',
  'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
  untar=True
)

batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset='training',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset='validation',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# Should be 5 class names.
class_names = np.array(train_ds.class_names)
print(class_names)

# Expect floats in [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Here x is the image, y is the label.
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# See what an image looks like
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Gather the labels
labels_path = tf.keras.utils.get_file(
  'ImageNetLabels.txt',
  'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
)
imagenet_labels = np.array(open(labels_path).read().splitlines())

# TODO: We should verify as much info as possible about any classifier we download.

# Strictly speaking it makes quite a bit of sense to download a high-level classifier rather than training our own.
# Training our own is expensive and also not the point.
# By downloading a "standard" one we are showing that we are highly performant on an actual use case.

# It also is a good sign that the MobileNetV2 stuff is from an actual paper from Google.
# However we still do need to verify it all.

# Define our classifier, a pre-trained one from tfhub.
mobilenet_v2 = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'
inception_v3 = 'https://tfhub.dev/google/imagenet/inception_v3/classification/5'

classifier_model = mobilenet_v2

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
  hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])

"""
# Use the existing classifier to predict on this dataset.
# Shouldn't work super well as the classifier was trained on different classes.
with tf.device(device):
  time_1 = time.time_ns()
  result_batch = classifier.predict(train_ds)
  time_2 = time.time_ns()

# Already here we are seeing something like 1.6ms / image on the GPU and 5.5ms / image on the CPU.
# I would expect these times will be even more improved by adding our own training layer.
print("Predicting:", (time_2 - time_1) / 1000000, "ms")

# Observe the predicted class names
predicted_class_names = imagenet_labels[tf.math.argmax(result_batch, axis=-1)]
print(predicted_class_names)
"""

# Let's instead use transfer learning; we'll get a model without the top layer and train our own top layer.
# This way we should get much better accuracy.
mobilenet_v2 = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
inception_v3 = 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4'

feature_extractor_model = mobilenet_v2

feature_extractor_layer = hub.KerasLayer(
  feature_extractor_model,
  input_shape=(224, 224, 3),
  trainable=False
)

# Returns a 1280-long vector (that's a lot of features.)
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

num_classes = len(class_names)

model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(num_classes)
])

print(model.summary())

# This way we get 1.8ms vs 6.4ms, which I guess is a slightly better ratio lmao.
# Now let's actually compile and train our model.
with tf.device(device):
  time_1 = time.time_ns()

  # Compiling the model.
  model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc']
  )

  time_2 = time.time_ns()

  # Possibly add the tensorboard / plt stuff later.

  # This is a relatively small number of epochs.
  NUM_EPOCHS = 10

  # Training the model.
  model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS
  )

  time_3 = time.time_ns()

  # Testing the model
  for image_batch, labels_batch in train_ds:
    predicted_batch = model.predict(image_batch)
    # predicted_id = tf.math.argmax(predicted_batch, axis=-1)
    # predicted_label_batch = class_names[predicted_id]

  time_4 = time.time_ns()

  print("Compiling:", (time_2 - time_1) / 1000000, "ms")
  print("Training:", (time_3 - time_2) / 1000000, "ms")
  print("Predicting:", (time_4 - time_3) / 1000000, "ms")










