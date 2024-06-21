from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Categories that the model predicts
categories = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax',
    'Normal'
]

# Path to the model weights file
weights_path = 'mobilenet_1_0_128_tf_no_top.h5'

# Check if the model weights file exists
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights file not found at {weights_path}")

# Load the base model
base_model = MobileNet(weights=weights_path, include_top=False, input_shape=(224, 224, 3))

# Add new top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(categories), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on your data (replace `train_data` and `validation_data` with your datasets)
# model.fit(train_data, validation_data=validation_data, epochs=10)

# Save the complete model
model.save('model.h5')
