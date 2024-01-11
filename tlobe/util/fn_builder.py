import tensorflow as tf
import tensorflow_hub as hub
import bert
from bert import optimization


def model_fn_builder(num_labels, learning_rate, num_train_steps, num_warmup_steps, model_handle):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, mode):
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        if not is_predicting:

            (loss, predicted_labels, log_probs) = create_model(is_predicting, input_ids, input_mask, segment_ids,
                                                               label_ids, num_labels, model_handle)

            train_op = bert.optimization.create_optimizer(loss, learning_rate, num_train_steps, num_warmup_steps,
                                                          use_tpu=False)

            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.compat.v1.metrics.accuracy(label_ids, predicted_labels)
                true_pos = tf.compat.v1.metrics.true_positives(
                    label_ids,
                    predicted_labels)
                true_neg = tf.compat.v1.metrics.true_negatives(
                    label_ids,
                    predicted_labels)
                false_pos = tf.compat.v1.metrics.false_positives(
                    label_ids,
                    predicted_labels)
                false_neg = tf.compat.v1.metrics.false_negatives(
                    label_ids,
                    predicted_labels)

                return {
                    "eval_accuracy": accuracy,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg,
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs, output_layer) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels, model_handle)
            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels,
                'pooled_output': output_layer
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    return model_fn


def input_fn_builder(features, seq_length, is_training, drop_remainder):
    all_input_ids = [list(map(int, f.input_ids)) for f in features]
    all_input_mask = [list(map(int, f.input_mask)) for f in features]
    all_segment_ids = [list(map(int, f.segment_ids)) for f in features]
    all_label_ids = [int(f.label_id) for f in features]

    def input_fn(params):
        batch_size = params['batch_size']
        num_examples = len(features)

        d = tf.data.Dataset.from_tensor_slices({
            'input_ids': tf.constant(all_input_ids, shape=(num_examples, seq_length), dtype=tf.int32),
            'input_mask': tf.constant(all_input_mask, shape=(num_examples, seq_length), dtype=tf.int32),
            'segment_ids': tf.constant(all_segment_ids, shape=(num_examples, seq_length), dtype=tf.int32),
            'label_ids': tf.constant(all_label_ids, shape=(num_examples,), dtype=tf.int32)
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        if batch_size is None:
            print("null batch size; defaulting to 32")
            batch_size = 32

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels, num_labels, model_handle):
    bert_module = hub.Module(model_handle, trainable=True)
    bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
    bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)

    output_layer = bert_outputs["pooled_output"]
    output_layer1 = bert_outputs["pooled_output"]
    hidden_size = output_layer.shape[-1]

    output_weights = tf.compat.v1.get_variable("output_weights", [num_labels, hidden_size],
                                               initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.compat.v1.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.compat.v1.variable_scope("loss"):
        output_layer = tf.nn.dropout(output_layer, rate=0.2)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        if is_predicting:
            return predicted_labels, log_probs, output_layer1

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return loss, predicted_labels, log_probs
