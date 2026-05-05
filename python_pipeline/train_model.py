import os
import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Rescaling
import numpy as np
import config as cfg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def load_datasets():
    print("=== Loading Datasets ===")
    train_dir = os.path.join(cfg.CLEAN_DATA_DIR, "train")
    val_dir   = os.path.join(cfg.CLEAN_DATA_DIR, "validation")
    test_dir  = os.path.join(cfg.CLEAN_DATA_DIR, "test")

    # Load raw [0,255] first — normalization happens AFTER augmentation
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir, image_size=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH),
        batch_size=cfg.BATCH_SIZE, label_mode='int'
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir, image_size=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH),
        batch_size=cfg.BATCH_SIZE, label_mode='int'
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir, image_size=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH),
        batch_size=cfg.BATCH_SIZE, label_mode='int'
    )

    print(f"Class names (label order): {train_ds.class_names}")
    return train_ds, val_ds, test_ds

def calculate_class_weights(train_dir):
    sorted_classes = sorted(cfg.CLASSES)
    
    class_counts = {}
    for class_id in cfg.CLASSES:
        folder_path = os.path.join(train_dir, class_id)
        if os.path.exists(folder_path):
            class_counts[class_id] = len([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
        else:
            class_counts[class_id] = 0

    class_weights = {}
    print("\n=== Class Distribution & Weights ===")

    for keras_idx, class_id in enumerate(sorted_classes):
        count = class_counts[class_id]

        if cfg.MANUAL_CLASS_WEIGHTS is not None:
            # Use manually specified weights
            weight = cfg.MANUAL_CLASS_WEIGHTS.get(keras_idx, 1.0)
        else:
            # Calculate automatically from data distribution
            max_count = max(class_counts.values())
            weight = max_count / count if count > 0 else 1.0

        class_weights[keras_idx] = weight
        print(f"  Folder '{class_id}' -> Keras label {keras_idx}: "
              f"{count} images -> Weight: {weight:.2f}")

    print("====================================\n")
    return class_weights

def build_augmentation():
    return keras.Sequential([
        keras.layers.GaussianNoise(0.05),
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.15),
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.RandomBrightness(0.3),
        keras.layers.RandomContrast(0.3),
    ])

def sparse_focal_loss(gamma=2.0, class_weights=None):
    """
    Focal loss for sparse (integer) labels.
    gamma=2.0 focuses training on hard misclassified examples.
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # One-hot encode
        y_true_one_hot = tf.one_hot(y_true, depth=cfg.NUM_CLASSES)

        # Cross entropy
        ce = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)

        # Focal weight: (1 - p_t)^gamma
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, gamma)

        loss = focal_weight * ce

        # Apply class weights if provided
        if class_weights is not None:
            weights_tensor = tf.constant(
                [class_weights[i] for i in range(cfg.NUM_CLASSES)],
                dtype=tf.float32
            )
            sample_weights = tf.reduce_sum(
                y_true_one_hot * weights_tensor, axis=-1
            )
            loss = loss * sample_weights

        return tf.reduce_mean(loss)
    return loss_fn


def build_model(class_weights=None):
    inputs = keras.Input(shape=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3))

    x = Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(cfg.NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

def count_images(directory):
    print(f"\n=== Image counts in {directory} ===")
    total = 0
    for class_id in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_id)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path)
                        if f.lower().endswith(('.jpg','.jpeg','.png'))])
            print(f"  Class {class_id}: {count} images")
            total += count
    print(f"  Total: {total} images")
    
def evaluate_per_class(model, test_ds):
    print("\n=== Per-Class Accuracy ===")
    all_preds = []
    all_labels = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    for class_id in range(cfg.NUM_CLASSES):
        mask = all_labels == class_id
        if mask.sum() == 0:
            print(f"  Class {class_id}: no samples")
            continue
        correct = (all_preds[mask] == class_id).sum()
        total   = mask.sum()
        print(f"  Class {class_id}: {correct}/{total} correct ({100*correct/total:.1f}%)")

        
def train_and_export():
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    import shutil
    old_export = os.path.join(cfg.MODEL_SAVE_DIR, 'saved_model_export')
    if os.path.exists(old_export):
        shutil.rmtree(old_export)
        print(f"[INFO] Cleared old SavedModel export")

    train_ds, val_ds, test_ds = load_datasets()
    AUTOTUNE = tf.data.AUTOTUNE

    augmentation = build_augmentation()

    def augment_and_normalize(images, labels):
        images = tf.cast(images, tf.float32)
        images = augmentation(images, training=True)
        images = images / 255.0
        return images, labels

    def normalize_only(images, labels):
        return tf.cast(images, tf.float32) / 255.0, labels

    train_ds_fit = (train_ds
        .map(augment_and_normalize, num_parallel_calls=AUTOTUNE)
        .shuffle(1000)
        .prefetch(AUTOTUNE))
    val_ds_fit   = val_ds.map(normalize_only).cache().prefetch(AUTOTUNE)
    test_ds_norm = test_ds.map(normalize_only)

    weights = calculate_class_weights(os.path.join(cfg.CLEAN_DATA_DIR, "train"))
    model = build_model()

    print("\n=== Training Model ===")
    keras_path = os.path.join(cfg.MODEL_SAVE_DIR, 'face_model.keras')
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            min_delta=0.005
        ),
        keras.callbacks.ModelCheckpoint(
            keras_path,
            monitor='val_accuracy',
            save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6
        )
    ]

    model.fit(
        train_ds_fit,
        validation_data=val_ds_fit,
        epochs=cfg.EPOCHS,
        class_weight=weights,
        callbacks=callbacks
    )

    print("\n=== Final Test Set Evaluation ===")
    test_loss, test_acc = model.evaluate(test_ds_norm)
    print(f"Overall Test Accuracy: {test_acc*100:.2f}%")
    evaluate_per_class(model, test_ds_norm)

    print("\n=== Converting to TFLite (INT8) ===")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("\n=== Converting to TFLite (INT8) ===")

    saved_model_path = os.path.join(cfg.MODEL_SAVE_DIR, 'saved_model_export')
    model.export(saved_model_path)
    print(f"[INFO] SavedModel exported to {saved_model_path}")

    # Fresh calibration dataset
    calib_ds = keras.utils.image_dataset_from_directory(
        os.path.join(cfg.CLEAN_DATA_DIR, "train"),
        image_size=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH),
        batch_size=16,
        label_mode='int',
        shuffle=False
    ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    sample_count = sum(b.shape[0] for b, _ in calib_ds)
    print(f"[INFO] Calibration samples: {sample_count}")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for images, _ in calib_ds:
            yield [images]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # Verify
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    ind  = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    print(f"[INFO] Input  scale={ind['quantization'][0]:.8f}  zero_point={ind['quantization'][1]}")
    print(f"[INFO] Output scale={outd['quantization'][0]:.8f}  zero_point={outd['quantization'][1]}")

    if ind['quantization'][0] == 0.0:
        print("[ERROR] Calibration FAILED — do not flash this model!")
    else:
        print("[OK] Calibration successful — safe to flash")

    tflite_path = os.path.join(cfg.MODEL_SAVE_DIR, 'face_model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    export_to_c_array(tflite_model)

def export_to_c_array(tflite_model):
    """Converts the TFLite binary into C++ code and dumps it in the ESP32 folder."""
    print(f"\n=== Exporting C++ Code to ESP32 Folder ===")
    os.makedirs(cfg.ESP32_MAIN_DIR, exist_ok=True)
    
    h_path = os.path.join(cfg.ESP32_MAIN_DIR, 'model_data.h')
    cc_path = os.path.join(cfg.ESP32_MAIN_DIR, 'model_data.cc')
    
    hex_lines = [', '.join([f'0x{b:02x}' for b in tflite_model[i:i+12]]) for i in range(0, len(tflite_model), 12)]
    hex_array = ',\n  '.join(hex_lines)

    with open(h_path, 'w') as f:
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        f.write("extern const unsigned char g_model[];\n")
        f.write("extern const int g_model_len;\n\n")
        f.write("#endif\n")

    with open(cc_path, 'w') as f:
        f.write("#include \"model_data.h\"\n\n")
        f.write(f"alignas(8) const unsigned char g_model[] = {{\n {hex_array}\n}};\n")
        f.write(f"const int g_model_len = {len(tflite_model)};\n")
        
    print(f"[OK] Saved model_data.h to {cfg.ESP32_MAIN_DIR}")
    print(f"[OK] Saved model_data.cc to {cfg.ESP32_MAIN_DIR}")