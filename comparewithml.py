import os,joblib
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras import layers, models

# Assuming you have a list of traces (list of lists) and labels
# traces_raw = [['cmd1', 'cmd2', ...], ['cmd3', ...], ...]
# labels = [0, 1, 0, ...]


headers=list('ABCD')
agents=[]
traces_raw = []
labels=[]
if 0:
    for f in os.listdir('/one/work_caches/ai_traces'):
        print('f ', f)
        df = pd.read_csv(f'/one/work_caches/ai_traces/{f}', header=None, names=headers)
        print('df.columns ', df.columns)
        for h in ['C','D']:
            agents.append(f'{f[:3]}_{h}')
            labels.append(1 if f.startswith(('h','b')) else 0)
            traces_raw.append(df[h].to_list())

    joblib.dump((agents,traces_raw,labels), 'ml_data.joblib')
    print('dumped')
else:
    agents,traces_raw,labels = joblib.load('ml_data.joblib')
    
raw_len=5000
traces_raw = [trace[:raw_len] for trace in traces_raw]

# Flatten & encode symbolic tokens
all_tokens = sorted({token for trace in traces_raw for token in trace})
token_to_int = {token: idx for idx, token in enumerate(all_tokens)}
vocab_size = len(token_to_int)

# Integer encode each trace
max_len = max(len(trace) for trace in traces_raw)
traces = np.array([
    [token_to_int.get(tok, 0) for tok in trace] + [0] * (max_len - len(trace))
    for trace in traces_raw
])


if 1:
    tracelen = 1024
    traces_test = traces[:, 1000:1000+tracelen]  # if enough length
    traces = np.array([
        [token_to_int.get(tok, 0) for tok in trace] + [0] * (tracelen - len(trace))
        for trace in traces_raw
    ])
    traces = traces[:, :tracelen]  # enforce shape
    labels = np.array(labels)
    print('labels.shape ', labels.shape)

    # Split data
    traces_train = traces  # use all for now, or split
    decoder_targets = np.expand_dims(traces_train, -1)

    # Model input
    input_dim = tracelen
    # ... model setup same ...
    embedding_dim = 32
    latent_dim = 64

    inputs = layers.Input(shape=(input_dim,))
    x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_dim)(inputs)
    x_flat = layers.Flatten()(x)
    encoded = layers.Dense(latent_dim, activation='relu')(x_flat)

    # Decoder (reconstruction)
    decoded = layers.Dense(input_dim * embedding_dim, activation='relu')(encoded)
    decoded = layers.Reshape((input_dim, embedding_dim))(decoded)
    decoded = layers.Dense(vocab_size, activation='softmax')(decoded)

    # Classifier
    classification = layers.Dense(32, activation='relu')(encoded)
    classification = layers.Dense(1, activation='sigmoid')(classification)

    # Model with two outputs
    model = models.Model(inputs=inputs, outputs=[decoded, classification])
    model.compile(optimizer='adam',
                  loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
                  loss_weights=[1.0, 0.5],
                  metrics={'dense_2': 'accuracy'})
    model.summary()

    model.fit(
        x=traces_train,
        y=[decoder_targets, labels],
        batch_size=32,
        epochs=10,
        validation_split=0.2
    )

    # Predict
    #recon, preds = model.predict(traces_test)
    # Only classification output
    _, preds = model.predict(traces_test)
    pred_labels = (preds > 0.5).astype(int)

    from sklearn.metrics import classification_report, confusion_matrix

    print("\n📊 Classification report:")
    print(classification_report(labels, pred_labels, target_names=["AtomicRedTeam", "angr"]))

    print("\n🧾 Confusion matrix:")
    print(confusion_matrix(labels, pred_labels))


else:
    # trim for memory mgmt
    tracelen = 256
    #traces_test = [trace[1000:1000+tracelen] for trace in traces]
    traces_test = traces[:, 1000:1000+tracelen]

    traces = [trace[:tracelen] for trace in traces]
    print('len(labels) ', len(labels))
    print('len(agents) ', len(agents))
    #labels = np.array(labels[:tracelen])
    labels = np.array(labels)


    #nput_dim = max_len  # length of trace
    input_dim = tracelen
    embedding_dim = 32
    latent_dim = 64

    inputs = layers.Input(shape=(input_dim,))
    x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_dim)(inputs)
    x_flat = layers.Flatten()(x)
    encoded = layers.Dense(latent_dim, activation='relu')(x_flat)

    # Decoder (reconstruction)
    decoded = layers.Dense(input_dim * embedding_dim, activation='relu')(encoded)
    decoded = layers.Reshape((input_dim, embedding_dim))(decoded)
    decoded = layers.Dense(vocab_size, activation='softmax')(decoded)

    # Classifier
    classification = layers.Dense(32, activation='relu')(encoded)
    classification = layers.Dense(1, activation='sigmoid')(classification)

    # Model with two outputs
    model = models.Model(inputs=inputs, outputs=[decoded, classification])
    model.compile(optimizer='adam',
                  loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
                  loss_weights=[1.0, 0.5],
                  metrics={'dense_2': 'accuracy'})
    model.summary()


    #train
    print('training')
    # Prepare labels for decoder: each token must be reshaped as (N, T, 1)

    if 1:
        decoder_targets = np.expand_dims(traces, -1)
        model.fit(
            x=traces,
            y=[decoder_targets, labels],
            batch_size=32,
            epochs=10,
            validation_split=0.2
        )
    else:
        decoder_targets = np.expand_dims(traces_train, -1)
        model.fit(
            x=traces_train,
            y=[decoder_targets, labels],
            batch_size=32,
            epochs=10,
            validation_split=0.2
        )

        # sometnih evaluatge
        recon, preds = model.predict(traces_test)
        pred_classes = (preds > 0.5).astype(int)
        accuracy = (pred_classes.flatten() == labels_test).mean()
        print(f"Test Accuracy: {accuracy:.2%}")

        #classification.
        model = models.Sequential([
            layers.Input(shape=(max_len,)),
            layers.Embedding(input_dim=vocab_size, output_dim=64),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(traces, labels, batch_size=32, epochs=10, validation_split=0.2)


