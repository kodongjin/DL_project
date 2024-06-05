import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

test_path = 'Img/test'

batch_size = 1024
AUTOTUNE = tf.data.experimental.AUTOTUNE

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    image_size=(100, 40),
    batch_size=batch_size,
    shuffle=False
)

class_names = test_dataset.class_names
print("Class names:")
for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")

test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = load_model('best_model.h5')

    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_accuracy}")

    y_pred = model.predict(test_dataset)
    y_pred_classes = np.argmax(y_pred, axis=1)

    y_true = np.concatenate([y for x, y in test_dataset], axis=0)

    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.savefig('visualization.png')