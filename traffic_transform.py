import tensorflow as tf

import traffic_constants

# Unpack constants
_DENSE_FLOAT_FEATURE_KEYS = traffic_constants.DENSE_FLOAT_FEATURE_KEYS
_RANGE_FEATURE_KEYS = traffic_constants.RANGE_FEATURE_KEYS
_VOCAB_FEATURE_KEYS = traffic_constants.VOCAB_FEATURE_KEYS
_CATEGORICAL_FEATURE_KEYS = traffic_constants.CATEGORICAL_FEATURE_KEYS
_VOLUME_KEY = traffic_constants.VOLUME_KEY
_transformed_name = traffic_constants.transformed_name


def preprocessing_fn(inputs):
    """
    Feature engineering pipeline using pure TensorFlow (no TFT)
    """
    outputs = {}

    # ==============================
    # 1. Z-score scaling (continuous features)
    # ==============================
    for key in _DENSE_FLOAT_FEATURE_KEYS:
        mean = tf.reduce_mean(inputs[key])
        std = tf.math.reduce_std(inputs[key])
        outputs[_transformed_name(key)] = (inputs[key] - mean) / (std + 1e-6)

    # ==============================
    # 2. Min-max scaling (range features)
    # ==============================
    for key in _RANGE_FEATURE_KEYS:
        min_val = tf.reduce_min(inputs[key])
        max_val = tf.reduce_max(inputs[key])
        outputs[_transformed_name(key)] = (inputs[key] - min_val) / (max_val - min_val + 1e-6)

    # ==============================
    # 3. NEW FEATURE: Temperature category
    # ==============================
    temp_mean = tf.reduce_mean(inputs['temp'])
    outputs['temp_category_xf'] = tf.cast(inputs['temp'] > temp_mean, tf.int64)

    # ==============================
    # 4. NEW FEATURE: Interaction
    # ==============================
    outputs['temp_cloud_interaction_xf'] = tf.cast(inputs['temp'], tf.float32) * tf.cast(inputs['clouds_all'], tf.float32)

    # ==============================
    # 5. Simple categorical encoding (hashing)
    # ==============================
    for key in _VOCAB_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tf.strings.to_hash_bucket_fast(inputs[key], 10)

    # ==============================
    # 6. Keep categorical features as-is
    # ==============================
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = inputs[key]

    # ==============================
    # 7. Label transformation
    # ==============================
    volume = tf.cast(inputs[_VOLUME_KEY], tf.float32)
    mean_volume = tf.reduce_mean(volume)
    outputs[_transformed_name(_VOLUME_KEY)] = tf.cast(volume > mean_volume, tf.int64)

    return outputs