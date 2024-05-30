# !!! conda activate mnist_env
from constants import *
from helper_functions import preprocess_and_feature_engineer
import tensorflow as tf
import pandas as pd
import nni
import tf2onnx

# Assuming FEATURE_COLUMNS and TARGET_COLUMNS are defined in your constants module or script
# FEATURE_COLUMNS = ["rorientw", "rorientx", "rorienty", "rorientz",  "reyedirz", "altitude", "dot_eyedir_sundir", "azimuth"]
# TARGET_COLUMNS = ['Cdx','Cdy','Cdz']

FILE_PATH = "attributes_iso.csv"
OUTPUT_FILE_PATH = "attributes_preprocessed.csv"
ONNX_OUTPUT_PATH = "model_Cd.onnx"

def has_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected")

def info():
    print("---> Running main_b.py <---\n")
    print("Features:\t", FEATURE_COLUMNS)
    print("Targets:\t", TARGET_COLUMNS)
    print("File:\t\t", FILE_PATH)
    print("Output File:\t", OUTPUT_FILE_PATH)

def load_dataset():
    data = pd.read_csv(OUTPUT_FILE_PATH)
    
    # Shuffle the dataset
    shuffled_data = data.sample(frac=1, random_state=41).reset_index(drop=True)
    
    features = shuffled_data[FEATURE_COLUMNS].values
    target = shuffled_data[TARGET_COLUMNS].values
    
    # Split data
    split_index = int(0.8 * len(data))
    x_train = features[:split_index]
    x_test = features[split_index:]
    y_train = target[:split_index]
    y_test = target[split_index:]
    
    return (x_train, y_train), (x_test, y_test)

def create_model(num_units, dropout_rate, lr, activation, num_layers, optimizer, weight_decay):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(len(FEATURE_COLUMNS),))
    ])

    # Add hidden layers
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(num_units, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    # Output layer
    model.add(tf.keras.layers.Dense(len(TARGET_COLUMNS), activation='linear')) 
 
    model.compile(
        loss='mean_squared_error',  # Use 'mean_squared_error' for regression, change if classification
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['mae']  # Mean Absolute Error for regression
    )
    return model

def train(params):
    class ReportIntResult(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            acc = logs.get('val_mae')  # Use appropriate metric for your task
            if acc:
                nni.report_intermediate_result(acc)

    num_units = params.get('num_units')
    dropout_rate = params.get('dropout_rate')
    lr = params.get('lr')
    activation = params.get('activation')
    batch_size = params.get('batch_size')
    num_layers = params.get('num_layers')
    optimizer = params.get('optimizer')
    weight_decay = params.get('weight_decay')

    model = create_model(num_units, dropout_rate, lr, activation, num_layers, optimizer, weight_decay)
    (x_train, y_train), (x_test, y_test) = load_dataset()

    # Create TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=35, restore_best_weights=True)
        
    # Save the best model
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor='val_mae', mode='min'
    )

    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=60,
        callbacks=[ReportIntResult(), early_stopping_callback, best_model_callback],  # Make sure to include callbacks here
        verbose=2
    )
    _, acc = model.evaluate(test_dataset, verbose=0)
    nni.report_final_result(acc)
    
    # Convert to ONNX format
    model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model, output_path=ONNX_OUTPUT_PATH)
    print(f"Model saved to {ONNX_OUTPUT_PATH}")

if __name__ == '__main__':
    info()
    preprocess_and_feature_engineer(FILE_PATH, OUTPUT_FILE_PATH, 100)
    (x_train, y_train), (x_test, y_test) = load_dataset()
    print(f"\nx_train shape (features): \t{x_train.shape}")
    print(f"y_train shape (targets): \t{y_train.shape}")
    
    params = {
        'num_units': 128,
        'dropout_rate': 0.451,
        'lr': 0.01,
        'activation': 'tanh',
        'batch_size': 64,
        'num_layers': 3,
        'optimizer': 'adam',
        'weight_decay': 1e-5
    }

    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)

    train(params)
    print("\n=== Done running main_b.py ===\n")
