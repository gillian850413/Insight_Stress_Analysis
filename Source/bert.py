import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization


class BertModel:
    def __init__(self,
                 label_list,
                 bert_model_hub,
                 batch_size=32,
                 learning_rate=2e-5,
                 num_train_epochs=3,
                 warmup_proportion=0.1,
                 save_checkpoints_steps=500,
                 save_summary_steps=100):

        self.label_list = label_list
        self.bert_model_hub = bert_model_hub
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.save_checkpoints_steps = save_checkpoints_steps
        self.save_summary_steps = save_summary_steps

    def create_input(self, data, data_column, label_column):
        # Use the InputExample class from BERT's run_classifier code to create examples from the data
        input_examples = data.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                               text_a=x[data_column],
                                                                               text_b=None,
                                                                               label=x[label_column]), axis=1)
        return input_examples

    def create_tokenizer(self):
        """Get the vocab file and casing info from the Hub module.
        """
        with tf.Graph().as_default():
            bert_module = hub.Module(self.bert_model_hub)
            token_info = bert_module(signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([token_info["vocab_file"],
                                                      token_info["do_lower_case"]])

        return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def get_max_len(self, data):
        data.str.split(' ')
        max_len = 0
        for i in range(len(data)):
            if len(data.iloc[i]) > max_len:
                max_len = len(data.iloc[i])
        return max_len

    def convert_input_examples(self, input_examples, max_len, tokenizer):
        # Convert train/test features to InputFeatures that BERT understands.
        bert_features = bert.run_classifier.convert_examples_to_features(input_examples,
                                                                         self.label_list,
                                                                         max_len,
                                                                         tokenizer)
        return bert_features

    def create_model(self, is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
        """Creates a classification model."""

        bert_module = hub.Module(self.bert_model_hub, trainable=True)
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)

        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_outputs" for token-level output.
        output_layer = bert_outputs["pooled_output"]

        hidden_size = output_layer.shape[-1].value

        # Create our own layer to tune for politeness data.
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable("output_bias",
                                      [num_labels],
                                      initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            # Dropout helps prevent overfitting
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            # Convert labels into one-hot encoding
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
            # If we're predicting, we want predicted labels and the probabiltiies.
            if is_predicting:
                return (predicted_labels, log_probs)

            # If we're train/eval, compute loss between predicted and actual label
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return loss, predicted_labels, log_probs

    def model_fn_builder(self, num_labels, learning_rate, num_train_steps, num_warmup_steps):
        """Returns `model_fn` closure for TPUEstimator.
            # model_fn_builder actually creates our model function
            # using the passed parameters for num_labels, learning_rate, etc.
        """

        def model_fn(features, mode):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]

            is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

            # TRAIN and EVAL
            if not is_predicting:
                loss, predicted_labels, log_probs = self.create_model(is_predicting, input_ids, input_mask,
                                                                      segment_ids, label_ids, num_labels)

                train_op = bert.optimization.create_optimizer(loss,
                                                              learning_rate,
                                                              num_train_steps,
                                                              num_warmup_steps,
                                                              use_tpu=False)

                # Calculate evaluation metrics.
                def metric_fn(label_ids, predicted_labels):
                    accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                    f1_score = tf.contrib.metrics.f1_score(label_ids, predicted_labels)
                    auc = tf.metrics.auc(label_ids, predicted_labels)
                    recall = tf.metrics.recall(label_ids, predicted_labels)
                    precision = tf.metrics.precision(label_ids, predicted_labels)
                    true_pos = tf.metrics.true_positives(label_ids, predicted_labels)
                    true_neg = tf.metrics.true_negatives(label_ids, predicted_labels)
                    false_pos = tf.metrics.false_positives(label_ids, predicted_labels)
                    false_neg = tf.metrics.false_negatives(label_ids, predicted_labels)

                    return {
                        "eval_accuracy": accuracy,
                        "f1_score": f1_score,
                        "auc": auc,
                        "precision": precision,
                        "recall": recall,
                        "true_positives": true_pos,
                        "true_negatives": true_neg,
                        "false_positives": false_pos,
                        "false_negatives": false_neg
                    }

                eval_metrics = metric_fn(label_ids, predicted_labels)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
                else:
                    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)
            else:
                (predicted_labels, log_probs) = self.create_model(is_predicting,
                                                                  input_ids,
                                                                  input_mask,
                                                                  segment_ids,
                                                                  label_ids,
                                                                  num_labels)

                predictions = {'probabilities': log_probs, 'labels': predicted_labels}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Return the actual model function in the closure
        return model_fn

    # predict on new sentences
    def get_prediction(self, in_sentences):
        labels = ["Non-Stress", "Stress"]
        labels_idx = [0, 1]
        input_examples = [run_classifier.InputExample(guid="", text_a=x, text_b=None, label=0) for x in in_sentences]
        # here, "" is just a dummy label

        input_features = run_classifier.convert_examples_to_features(input_examples,
                                                                     labels_idx,
                                                                     max_seq_len,
                                                                     tokenizer)

        predict_input_fn = run_classifier.input_fn_builder(features=input_features,
                                                           seq_length=max_seq_len,
                                                           is_training=False,
                                                           drop_remainder=False)

        predictions = estimator.predict(predict_input_fn)
        return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in
                zip(in_sentences, predictions)]


if __name__ == "__main__":
    # load the data
    path = '/Users/gillianchiang/Desktop/Insight/Project/Insight_Stress_Analysis/data/'
    train = pd.read_csv(path + 'dreaddit-train.csv', encoding="ISO-8859-1")
    test = pd.read_csv(path + 'dreaddit-test.csv', encoding="ISO-8859-1")

    # define bert model object
    bert_model_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    bert_model = BertModel(label_list=[0, 1], bert_model_hub=bert_model_hub)

    # preprocess the data and convert the features into bert-readable format
    input_examples = bert_model.create_input(data=train, data_column="text", label_column="label")
    tokenizer = bert_model.create_tokenizer()
    max_seq_len = bert_model.get_max_len(train['text'])
    bert_input_ft = bert_model.convert_input_examples(input_examples, max_seq_len, tokenizer)

    # define the training and warm up steps
    num_train_steps = int(len(bert_input_ft) / bert_model.batch_size * bert_model.num_train_epochs)
    num_warmup_steps = int(num_train_steps * bert_model.warmup_proportion)

    # create an model checkpoint output directory
    path = os.getcwd()
    output_dir = os.path.join(path, "Framework/Output")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    run_config = tf.estimator.RunConfig(model_dir=output_dir,
                                        save_summary_steps=bert_model.save_summary_steps,
                                        save_checkpoints_steps=bert_model.save_checkpoints_steps)

    model_fn = bert_model.model_fn_builder(num_labels=len(bert_model.label_list),
                                           learning_rate=bert_model.learning_rate,
                                           num_train_steps=num_train_steps,
                                           num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       config=run_config,
                                       params={"batch_size": bert_model.batch_size})

    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(features=bert_input_ft,
                                                          seq_length=max_seq_len,
                                                          is_training=True,
                                                          drop_remainder=False)
    # train the model
    print(f'Beginning Training!')
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", datetime.now() - current_time)

    # test the model
    # preprocess test data
    test_input_examples = bert_model.create_input(data=test, data_column="text", label_column="label")
    test_input_ft = bert_model.convert_input_examples(test_input_examples, max_seq_len, tokenizer)

    test_input_fn = run_classifier.input_fn_builder(features=test_input_ft,
                                                    seq_length=max_seq_len,
                                                    is_training=False,
                                                    drop_remainder=False)

    # evaluate the model
    print(estimator.evaluate(input_fn=test_input_fn, steps=None))

    # predict new sentences
    pred_sentences = [
        "That movie was absolutely awful",
        "The acting was a bit lacking",
    ]

    predictions = bert_model.get_prediction(pred_sentences)
    print(predictions)

