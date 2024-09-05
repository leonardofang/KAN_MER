import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import optimizers, losses
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
# from fastkan_attention import Attent_FastKAN
from fastkan import FastKAN
import time

import tensorflow as tf
from tensorflow.keras import layers

# Load labels
label_path = '' # the dataset source, such as 'dataset/IEMOCAP_full_release/combined_labels.npy.npz'
labels = np.load(label_path, allow_pickle=True)['arr_0']

# Load data
data_path = '' # 'dataset/IEMOCAP_full_release/reduced_features.npy.npz'
data = np.load(data_path, allow_pickle=True)['arr_0']

# Ensure the data and labels have the same length
assert len(data) == len(labels), "Data and labels must have the same number of elements."

# Define the number of classes
num_classes = len(np.unique(labels))

# Convert labels to integer encoding if necessary (assuming labels are categorical)
if labels.ndim > 1:
    labels = np.argmax(labels, axis=1)

# Set up 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

accuracy_list = []
f1_list = []

fold = 1
for train_index, test_index in kf.split(data):
    print(f"Fold {fold} / 10")

    # Split data into training and validation based on current fold
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Create TensorFlow datasets
    batch_size = 32
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1024).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    # Define model
    model = FastKAN([X_train.shape[1], 4, num_classes])  # Adjust output layer to the number of classes

    # Define optimizer, loss, and accuracy metric
    optimizer = optimizers.legacy.Adam(learning_rate=1e-3)
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)  # Use SparseCategoricalCrossentropy for integer labels
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # Iterate over the batches of the dataset.
    for epoch in range(20):  # Assuming you want to train for several epochs per fold
        print(f'Start of epoch {epoch+1}')

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            accuracy_metric.update_state(y_batch_train, logits)
            if step % 200 == 0:
                print(f'Training loss (for one batch) at step {step}: {float(loss_value)}')

        # Display metrics at the end of each epoch.
        train_acc = accuracy_metric.result()
        print(f'Training acc over epoch: {float(train_acc)}')

        # Reset training metrics at the end of each epoch
        accuracy_metric.reset_state()

        # Validation
        test_time_epoch = time.time()
        for x_batch_val, y_batch_val in test_dataset:
            val_logits = model(x_batch_val)
            val_loss_value = loss_fn(y_batch_val, val_logits)
            accuracy_metric.update_state(y_batch_val, val_logits)

        val_acc = accuracy_metric.result()
        # Predict and calculate F1 score
        predict = np.argmax(model.predict(X_test), axis=1)
        y = f1_score(y_test, predict, average='weighted')
        accuracy_metric.reset_state()

        accuracy_list.append(float(val_acc))
        f1_list.append(float(y))
        print(f'Validation acc: {float(val_acc)}')
        print(f'Validation f1: {float(y)}')

    fold += 1

# Print overall statistics after cross-validation
print("Max Accuracy: %.5f, Average Accuracy: %.5f" % (max(accuracy_list), sum(accuracy_list) / len(accuracy_list)))
print("Max F1-score: %.5f, Average F1-score: %.5f" % (max(f1_list), sum(f1_list) / len(f1_list)))
