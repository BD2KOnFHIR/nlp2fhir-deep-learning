import tensorflow as tf
from keras import backend as K

# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
from nltk.corpus import stopwords
from pathlib import Path
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk import word_tokenize
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Activation, LSTM, GRU
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras import optimizers

# set parameters:
maxlen = 500
batch_size = 32
embedding_dims = 20
filters = 200
kernel_size = 3
epochs = 40




diseases = ['Asthma', 'CAD', 'CHF',
            'Depression', 'Diabetes', 'GERD' , 'Gallstones', 'Hypercholesterolemia',
    'Hypertension', 'Hypertriglyceridemia', 'OA', 'OSA', 'Obesity', 'PVD', 'Venous_Insufficiency'
]



# Function to create model, required for KerasClassifier
def create_cnn_model(output_dim, max_features):
    # create model
    model = Sequential()
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    model.add(Dropout(0.2))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))

    # we use max pooling:
    model.add(GlobalMaxPooling1D())
    model.add(Dense(output_dim, activation='sigmoid'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'], )
    return model


# Function to create model, required for KerasClassifier
def create_lstm_model(output_dim, max_features):
    # create model
    model = Sequential()

    model.add(Embedding(max_features, 128))
    model.add(GRU(128, dropout=0.2))
    model.add(Dense(output_dim, activation='sigmoid'))
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    
    return model


tokenizer = Tokenizer(num_words=5000)

# # set directries based on run-time environment
# if in_docker == 'True':
#     model_dir = '/data/models/'
#     data_dir = '/data/data/'
# else:
model_dir = '/infodev1/non-phi-data/yanshan/dl4ehr/'
data_dir = 'data/'


# get model and convert to w2v
glove_input_file = model_dir + 'glove.6B.100d.txt'

word2vec_output_file = '/local2/tmp/w2v.txt'
glove2word2vec(glove_input_file, word2vec_output_file)
wv_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

stop_words = set(stopwords.words('english'))

le = LabelEncoder()


def get_input_seq(line):
    word_list = word_tokenize(line)
    word_list = [word.lower() for word in word_list if word.lower() not in stop_words]
    idx_seq = []

    for word in word_list:
        if wv_model.vocab.get(word):
            idx = wv_model.vocab.get(word).index
            idx_seq.append(idx)

    return idx_seq


# print_model_summary = True


def run_comorbidity(dataset, comorbidity, flag='rt'):
    data_flag = f'{dataset}_rt_{comorbidity}'

    input_texts = []
    with open('data/corpus/{}.clean.txt'.format(data_flag), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            input_texts.append(get_input_seq(line))

    train_seq = []
    test_seq = []
    train_labels = []
    test_labels = []
    total_labels = []

    with open('data/{}.txt'.format(data_flag), 'r') as f:
        lines = f.readlines()
        assert len(lines) == len(input_texts)
        for encode_seq, line in zip(input_texts, lines):
            line = line.strip()
            splits = line.split('\t')
            total_labels.append(splits[2])

            if splits[1] == 'train':
                train_seq.append(encode_seq)
                train_labels.append(splits[2])
            else:
                test_seq.append(encode_seq)
                test_labels.append(splits[2])

    le.fit(total_labels)

    X_train = sequence.pad_sequences(train_seq, maxlen=maxlen)
    y_train = le.fit_transform(train_labels)

    X_test = sequence.pad_sequences(test_seq, maxlen=maxlen)
    y_test = le.transform(test_labels)

    # model = create_cnn_model(len(le.classes_), max(X_train.max(), X_test.max()) + 1)
    model = create_lstm_model(len(le.classes_), max(X_train.max(), X_test.max()) + 1)

    # if print_model_summary:
    #     model.summary()
    #     print_model_summary = False

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        shuffle=False,
                        callbacks=[EarlyStopping(monitor='val_loss', verbose=1, patience=5)],
                        verbose=0,
                        validation_data=(X_test, y_test),
                        batch_size=batch_size)

    y_pred = model.predict(X_test)
    # get labels for predictions
    # target_names = [encoder.classes_[idx] for idx in set(y_test_idx)]

    # print(classification_report(y_test, y_pred.argmax(axis=1), target_names=le.classes_, digits=3))
    results_micro = precision_recall_fscore_support(y_test, y_pred.argmax(axis=1), average='micro')
    results_macro = precision_recall_fscore_support(y_test, y_pred.argmax(axis=1), average='macro')
    print(dataset, comorbidity)
    print(results_macro)
    return results_micro, results_macro


if __name__ == '__main__':

    for dataset in ['mimic', 'i2b2']:
        for flag in ['rt']:
            with open(f'results/obesity/{dataset}_lstm_{flag}_f1_macro.csv', 'w') as fo:
                for disease in diseases:
                    res_micro, res_macro = run_comorbidity(dataset, disease, flag)
                    fo.write(f'{dataset},{flag},{disease},{res_micro[0]},{res_macro[0]},{res_macro[1]},{res_macro[2]}\n')