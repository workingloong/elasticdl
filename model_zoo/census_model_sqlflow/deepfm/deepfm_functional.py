import itertools
import tensorflow as tf
import tensorflow.keras.backend as K

from model_zoo.census_model_sqlflow.feature_configs import (
    INPUT_SCHEMAS,
    LABEL_KEY,
    age_bucketize,
    capital_gain_bucketize,
    capital_loss_bucketize,
    education_hash,
    group1,
    group2,
    group3,
    hours_per_week_bucketize,
    marital_status_lookup,
    native_country_hash,
    occupation_hash,
    race_lookup,
    relationship_lookup,
    sex_lookup,
    workclass_lookup,
)
from model_zoo.census_model_sqlflow.keras_process_layers import (
    FingerPrint,
    Lookup,
    Concat,
    NumericBucket,
)


# The model definition from model zoo. It's functional style.
# Input Params:
#   input_layers: The input layers dict of feature inputs
#   input_tensors: list of integer tensors
def deepfm_classifier(input_layers, input_tensors, group_dims):
    # Wide Part
    wide_embeddings = []
    for i, tensor in enumerate(input_tensors[0:-1]):
        embedding = tf.keras.layers.Embedding(group_dims[i], 1)(tensor)
        embedding_sum = tf.keras.backend.sum(embedding, axis=1)
        wide_embeddings.append(embedding_sum)

    deep_embeddings = []
    for i, tensor in enumerate(input_tensors[0:-1]):
        embedding = tf.keras.layers.Embedding(group_dims[i], 8)(tensor)
        embedding_sum = tf.keras.backend.sum(embedding, axis=1)
        deep_embeddings.append(embedding_sum)

    group_num = len(deep_embeddings)
    embeddings = tf.concat(deep_embeddings, 1)  # shape = (None, group_num , 8)
    embeddings = tf.reshape(
        embeddings, shape=(-1, group_num, 8)
    )
    emb_sum = K.sum(embeddings, axis=1)  # shape = (None, 8)
    emb_sum_square = K.square(emb_sum)  # shape = (None, 8)
    emb_square = K.square(embeddings)  # shape = (None, group_num, 8)
    emb_square_sum = K.sum(emb_square, axis=1)  # shape = (None, 8)
    second_order = 0.5 * tf.keras.layers.Subtract()(
        [emb_sum_square, emb_square_sum]
    )

    first_order = tf.keras.layers.Concatenate()(wide_embeddings)

    # Deep Part
    dnn_input = tf.keras.layers.Concatenate()(deep_embeddings)
    for i in [16, 8, 4]:
        dnn_input = tf.keras.layers.Dense(i)(dnn_input)

    # Output Part
    print(first_order)
    print(second_order)
    print(dnn_input)
    concat_input = tf.concat([first_order, second_order, dnn_input], 1)

    logits = tf.reduce_sum(concat_input, 1, keepdims=True)
    probs = tf.reshape(tf.sigmoid(logits), shape=(-1,))

    return tf.keras.Model(
        inputs=input_layers,
        outputs={"logits": logits, "probs": probs},
        name="wide_deep",
    )


# Build the input layers from the schema of the input features
def get_input_layers(input_schemas):
    input_layers = {}

    for schema_info in input_schemas:
        input_layers[schema_info.name] = tf.keras.layers.Input(
            name=schema_info.name, shape=(1,), dtype=schema_info.dtype
        )

    return input_layers


# It can be generated from the parsed meta in feature_configs using code_gen.
def transform(source_inputs):
    inputs = source_inputs.copy()

    education_hash_out = FingerPrint(education_hash.param)(
        inputs["education"]
    )
    occupation_hash_out = FingerPrint(occupation_hash.param)(
        inputs["occupation"]
    )
    native_country_hash_out = FingerPrint(native_country_hash.param)(
        inputs["native-country"]
    )
    workclass_lookup_out = Lookup(workclass_lookup.param)(
        inputs["workclass"]
    )
    marital_status_lookup_out = Lookup(marital_status_lookup.param)(
        inputs["marital-status"]
    )
    relationship_lookup_out = Lookup(relationship_lookup.param)(
        inputs["relationship"]
    )
    race_lookup_out = Lookup(race_lookup.param)(inputs["race"])
    sex_lookup_out = Lookup(sex_lookup.param)(inputs["sex"])
    age_bucketize_out = NumericBucket(age_bucketize.param)(inputs["age"])
    capital_gain_bucketize_out = NumericBucket(capital_gain_bucketize.param)(
        inputs["capital-gain"]
    )
    capital_loss_bucketize_out = NumericBucket(capital_loss_bucketize.param)(
        inputs["capital-loss"]
    )
    hours_per_week_bucketize_out = NumericBucket(
        hours_per_week_bucketize.param
    )(inputs["hours-per-week"])

    group1_offsets = list(
        itertools.accumulate([0] + group1.param[:-1])
    )
    group1_out = Concat(group1_offsets)(
        [
            workclass_lookup_out,
            hours_per_week_bucketize_out,
            capital_gain_bucketize_out,
            capital_loss_bucketize_out,
        ]
    )
    group2_offsets = list(
        itertools.accumulate([0] + group2.param[:-1])
    )
    group2_out = Concat(group2_offsets)(
        [
            education_hash_out,
            marital_status_lookup_out,
            relationship_lookup_out,
            occupation_hash_out,
        ]
    )
    group3_offsets = list(
        itertools.accumulate([0] + group3.param[:-1])
    )
    group3_out = Concat(group3_offsets)(
        [
            age_bucketize_out,
            sex_lookup_out,
            race_lookup_out,
            native_country_hash_out,
        ]
    )
    group_ids = [group1_out, group2_out, group3_out]
    group_max_ids = [sum(group1.param), sum(group2.param), sum(group3.param)]
    print("group_max_ids", group_max_ids)
    return group_ids, group_max_ids


# The entry point of the submitter program
def custom_model():
    input_layers = get_input_layers(input_schemas=INPUT_SCHEMAS)
    group_ids, group_max_ids = transform(input_layers)

    return deepfm_classifier(
        input_layers, group_ids, group_max_ids
    )


def loss(labels, predictions):
    logits = predictions["logits"]
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(tf.reshape(labels, (-1, 1)), tf.float32),
            logits=logits,
        )
    )


def optimizer(lr=0.001):
    return tf.keras.optimizers.Adam(learning_rate=lr)


def eval_metrics_fn():
    return {
        "logits": {
            "accuracy": lambda labels, predictions: tf.equal(
                tf.cast(tf.reshape(predictions, [-1]) > 0.5, tf.int32),
                tf.cast(tf.reshape(labels, [-1]), tf.int32),
            )
        },
        "probs": {"auc": tf.keras.metrics.AUC()},
    }


def learning_rate_scheduler(model_version):
    if model_version < 5000:
        return 0.0003
    elif model_version < 12000:
        return 0.0002
    else:
        return 0.0001


def dataset_fn(dataset, mode, _):
    def _parse_data(record):
        feature_description = {}
        for schema in INPUT_SCHEMAS:
            feature_description[schema.name] = tf.io.FixedLenFeature(
                (1,), schema.dtype
            )
        feature_description[LABEL_KEY] = tf.io.FixedLenFeature([], tf.int64)

        print(feature_description)
        parsed_record = tf.io.parse_single_example(record, feature_description)
        label = parsed_record.pop(LABEL_KEY)

        return parsed_record, label

    dataset = dataset.map(_parse_data)

    return dataset
