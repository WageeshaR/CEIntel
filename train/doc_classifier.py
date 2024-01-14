import tensorflow as tf
from datetime import datetime
from official.nlp.data import classifier_data_lib
from preprocess.document import *
from train.util.fn_builder import model_fn_builder, input_fn_builder
from train.util.tokenization import create_tokenizer_from_hub_module

np.random.seed(1337)
OUTPUT_DIR = './output'
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


def bert_trainer(train, test):
    try:
        tf.io.gfile.rmtree(OUTPUT_DIR)
    except:
        pass

    tf.io.gfile.makedirs(OUTPUT_DIR)
    print('***** Model output directory: {} *****'.format(OUTPUT_DIR))
    label_list = [x for x in np.unique(train.label)]

    train_l = []
    label_l = []
    index_l = []
    for idx, row in train.iterrows():
        train_l.append(row['text'])
        label_l.append(row['label'])
        index_l.append(idx)

    test_l = []
    test_label_l = []
    test_index_l = []
    for idx, row in test.iterrows():
        test_l.append(row['text'])
        test_label_l.append(row['label'])
        test_index_l.append(idx)

    train_df = pd.DataFrame({'text': train_l, 'label': label_l})
    test_df = pd.DataFrame({'text': test_l, 'label': test_label_l})

    train_InputExamples = train_df.apply(
        lambda x: classifier_data_lib.InputExample(guid=None, text_a=x['text'], text_b=None, label=x['label']), axis=1)
    test_InputExamples = test_df.apply(
        lambda x: classifier_data_lib.InputExample(guid=None, text_a=x['text'], text_b=None, label=x['label']), axis=1)

    tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_HUB)

    train_features = []
    test_features = []
    for i in range(train_InputExamples.size):
        train_features.append(
            classifier_data_lib.convert_single_example(i, train_InputExamples[i], label_list, 200, tokenizer))

    for i in range(test_InputExamples.size):
        test_features.append(
            classifier_data_lib.convert_single_example(i, test_InputExamples[i], label_list, 200, tokenizer))

    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 1.0
    WARMUP_PROPORTION = 0.1
    SAVE_CHECKPOINTS_STEPS = 300
    SAVE_SUMMARY_STEPS = 100

    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    run_config = tf.estimator.RunConfig(model_dir=OUTPUT_DIR, save_summary_steps=SAVE_SUMMARY_STEPS,
                                        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    model_fn = model_fn_builder(num_labels=len(label_list), learning_rate=LEARNING_RATE,
                                num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps,
                                model_handle=BERT_MODEL_HUB)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params={"batch_size": BATCH_SIZE})

    train_input_fn = input_fn_builder(features=train_features, seq_length=200, is_training=True, drop_remainder=False)

    test_input_fn = input_fn_builder(features=test_features, seq_length=200, is_training=False, drop_remainder=False)

    print(f'Beginning Training!')
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", datetime.now() - current_time)


def train_and_evaluate(model_type, input_type, text_column, label_column, filename=None):
    data = None

    if input_type == "file":
        if filename is None:
            raise ValueError("filename must not be null")
        data = read_from_file(filename=filename)
    elif input_type == "db":
        # TODO: implement database retrieval
        return None

    if data is None:
        raise ValueError(f"Could not read data from {input_type}")

    data = null_filter(data, text_column)
    data = len_filter(data, text_column, 200)
    train_data, test_data = label_encoder(data, text_column, label_column)

    try:
        bert_trainer(train_data, test_data)
    except Exception as e:
        print(f'Error occurred during training of {model_type}: {e}')
