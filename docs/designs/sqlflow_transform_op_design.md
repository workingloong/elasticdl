# Design Of Transformation Operator in SQLFlow

This document describes the design for generating Python code using Tensorflow for transformation operators defined in SQLFlow.

## Motivation
According [data analysis and transformation](https://github.com/sql-machine-learning/elasticdl/blob/develop/docs/designs/data_transform.md#generate-transform-code-from-sqlflow-statement), users can define transformation keyword in SQLFlow. We need to convert those transformation keywords to executable Python codes using Tensorflow. Then we can transform the data to tensors for model during training and prediction. In ElasticDL, we use Keras API to define model. To implement transformation operators, we can:
1. Use `tf.feature_column` to transform data and use `DenseFeature` in model to receive the result of `tf.feature_column`.
2. Use Keras layers to transform data and the output of transformation layer can directly be used by other layers.

The following, the design will introduce the two implementations for each transformation operator.

## Transformation Operator

| Name | DESCRIPTION |
| ---- | -------------- | 
| NORMALIZE | Scaling inputs by removing the min and scaling to the max |
| STANDARDIZE | Standardize inputs by removing the mean and scaling to unit variance|
| BUCKETIZE | Discretized inputs to ID bucketed by boundaries.|
| HASH_BUCKET | Distribute the features into a finite number of buckets by hashing. output_id = Hash(input) % bucket_size for string type input |
| VOCABULARIZE | Map inputs to an integer ID according to a vocabulary |
| CONCAT | Concatenate multiple inputs to a dense tensor |
| CROSS | The transformation can be thought of as: Hash(cartesian product of features) % hash_bucket_size |
| EMBEDDING | Use this when your inputs are sparse, but you want to convert them to a dense representation |

### Implementation of Transformation Operator

1. NORMALIZE
The NORMALIZE can scale inputs to [0, 1] by (x - min) / max.

NORMALIZE(x, min, max) \
Arguments: \
    x: the input values \
    min: the minimum of the feature in dataset. \
    max: the maximum of the feature in dataset. \

Implementation the NORMALIZE using Keras:
```python
class Normalize(tf.keras.layers.Layer):
    """Scaling inputs by (x - min) / max
    Args:
        min: minimum value of all feature values
        max: maximum value of all feature values
    """
    def __init__(self, min, max):
        super(Normalize, self).__init__()
        self.min = min
        self.max = max

    def call(self, inputs):
        return (inputs - min) / max
```

Implementation the NORMALIZE using feature column:
```python

def generate_normalize_column(name, min, max)
    """
    Args:
        name: feature name
        min: minimum value of all feature values
        max: maximum value of all feature values
    """
    def normalize(x, min, max):
        return (x - min) / max

    transform_fn = lambda x, min=min, max=max: (normalize(x, min, max)
    return tf.feature_column.numeric_column(
        name,
        dtype=tf.float32,
        normalizer_fn=transform_fn
    )
```

The use in SQLFlow:
```sql
SELECT *
FROM census_income
TO TRAIN DNNClassifier
WITH model.hidden_units = [10, 20]
COLUMN (
    NORMALIZE(age), 
    NORMALIZE(capital_gain), 
    NORMALIZE(capital_loss), 
    NORMALIZE(hours_per_week),
)
LABEL label
```

```python
# We can acquire the features from SQLFlow COLUMN expression
NORMALIZE_FEATURES = ['age', 'captital_gain', 'capital_loss', 'hours_per_week']

# We suppose the min=0 and max=100 for all features.
# The min and max can be acquired by analysis or configured by users.
MIN = 0
MAX = 100

def get_input_layers():
    input_layers = {}
    for name in NORMALIZE_FEATURES:
        input_layers[feature] = tf.keras.layers.Input(
            name=name, shape=(1,), dtype=feature_info.dtype
        )
    return input_layers

```

```python
def transform(input_layers):
    transform_results = []
    for name in NORMALIZE_FEATURES:
        normalize = Normalize(MIN, MAX)(input_layers[name])
        transform_results.append(normalize)
    return transform_results

def dnn_model_keras(inputs, input_layers):
    concat = tf.keras.layers.Concatenate()(inputs)
    dense_1 = tf.keras.layers.Dense(16, activation="relu")(inputs)
    dense_2 = tf.keras.layers.Dense(16, activation="relu")(dense_1)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(dense_2)
    return tf.keras.Model(
        inputs=input_layers,
        outputs=output
    )

def custom_model():
    input_layers = get_input_layers()
    transform_result = transform(input_layers)
    return dnn_model_keras(transform_result, input_layers)
```

Feature column
```python
def normalize(x, min, max):
    return (x - min) / max

def get_feature_column():
    feature_columns = []
    for name in NORMALIZE_FEATURES:
        feature_columns.append(generate_normalize_column(name, MIN, MAX))
    return feature_columns

def dnn_model_feature_column(feature_columns, input_layers):
    dense_1 = tf.keras.layers.DenseFeatures(feature_columns)(input_layers)
    dense_1 = tf.keras.layers.Dense(16, activation="relu")(inputs)
    dense_2 = tf.keras.layers.Dense(16, activation="relu")(dense_1)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(dense_2)
    return tf.keras.Model(
        inputs=input_layers,
        outputs=output
    )

def custom_model():
    input_layers = get_input_layers()
    transform_result = transform(input_layers)
    return dnn_model_feature_column(transform_result, input_layers)
```
