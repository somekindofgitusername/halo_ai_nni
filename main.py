import tensorflow as tf
import nni

def has_gpu():
    # Check if GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to avoid TensorFlow allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected")

def load_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train/255., y_train), (x_test/255., y_test)

def create_model(num_units, dropout_rate, lr, activation):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_units, activation=activation),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['accuracy']
    )
    return model

def train(params):
    class ReportIntResult(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            acc = logs.get('val_accuracy')
            if acc:
                nni.report_intermediate_result(acc)

    num_units = params.get('num_units')
    dropout_rate = params.get('dropout_rate')
    lr = params.get('lr')
    activation = params.get('activation')
    batch_size = params.get('batch_size')

    model = create_model(num_units, dropout_rate, lr, activation)
    (x_train, y_train), (x_test, y_test) = load_dataset()

    _ = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=batch_size,
        callbacks=[ReportIntResult()],
        verbose=False
    )
    _, acc = model.evaluate(x_test, y_test, verbose=False)
    #print('Validation Acc:', acc)
    nni.report_final_result(acc)

if __name__=='__main__':
    has_gpu()
    params = {
        'num_units': 32,
        'dropout_rate': 0.1,
        'lr':0.001,
        'activation': 'relu',
        'batch_size': 1024
    }

    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)


    train(params)