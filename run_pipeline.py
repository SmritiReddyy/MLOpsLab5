import tensorflow as tf
from traffic_transform import preprocessing_fn
from testing_values import raw_data

# Convert raw_data into tensor format
def convert_to_tensor(data):
    tensor_data = {}
    for key in data[0]:
        values = [d[key] for d in data]

        if isinstance(values[0], str):
            tensor_data[key] = tf.constant(values)
        else:
            tensor_data[key] = tf.constant(values, dtype=tf.float32)

    return tensor_data


inputs = convert_to_tensor(raw_data)

# Run preprocessing function
outputs = preprocessing_fn(inputs)

print("\n=== TRANSFORMED OUTPUT ===\n")

for key, value in outputs.items():
    print(f"{key}: {value.numpy()}")