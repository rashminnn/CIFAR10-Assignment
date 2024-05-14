from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import seaborn as sns
import numpy as np
from dataclasses import dataclass


(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

plt.figure(figsize=(15, 6))
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i])
    plt.title(class_names[Y_train[i][0]])
    plt.axis('off')
plt.show()

X_temp, X_test, Y_temp, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 32*32*3)).reshape(-1, 32, 32, 3)
X_val_scaled = scaler.transform(X_val.reshape(-1, 32*32*3)).reshape(-1, 32, 32, 3)
X_test_scaled = scaler.transform(X_test.reshape(-1, 32*32*3)).reshape(-1, 32, 32, 3)

@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES: int = 10
    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 32
    NUM_CHANNELS: int = 3

@dataclass(frozen=True)
class TrainingConfig:
    EPOCHS: int = 20
    BATCH_SIZE: int = 256
    LEARNING_RATE: float = 0.002

def cnn_model_dropout():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    return model

model_dropout = cnn_model_dropout()
model_dropout.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=TrainingConfig.LEARNING_RATE), 
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_dropout.summary()

history = model_dropout.fit(X_train_scaled, Y_train, 
                            batch_size=TrainingConfig.BATCH_SIZE, 
                            epochs=TrainingConfig.EPOCHS, 
                            verbose=1, 
                            validation_data=(X_val_scaled, Y_val))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_loss, test_accuracy = model_dropout.evaluate(X_test_scaled, Y_test, verbose=0)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")
print(f"Training accuracy: {history.history['accuracy'][-1]}")

y_pred = model_dropout.predict(X_test_scaled)
y_pred_labels = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(Y_test, y_pred_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

precision = precision_score(Y_test, y_pred_labels, average='weighted')
recall = recall_score(Y_test, y_pred_labels, average='weighted')
print(f"Precision: {precision}")
print(f"Recall: {recall}")

learning_rates = [0.0001, 0.001, 0.01, 0.1]
train_losses = []
val_losses = []

for lr in learning_rates:
    print(f"Training model with learning rate: {lr}")
    model = cnn_model_dropout()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_scaled, Y_train,
                        batch_size=TrainingConfig.BATCH_SIZE, 
                        epochs=TrainingConfig.EPOCHS, 
                        validation_data=(X_val_scaled, Y_val), 
                        verbose=1)
    
    train_losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])

fig, axs = plt.subplots(2, 2, figsize=(14, 12))

for i, (lr, ax) in enumerate(zip(learning_rates, axs.flat)):
    ax.plot(train_losses[i], label=f'Training Loss (LR={lr})')
    ax.plot(val_losses[i], label=f'Validation Loss (LR={lr})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Learning Rate = {lr}')
    ax.legend()

plt.tight_layout()
plt.show()
