import time


def train_new_tax_model(n_lab, fn_temp, training_sequence, validation_sequence):
    from tensorflow.keras.layers import Dense
    from tensorflow.keras import Input
    from tensorflow.keras.layers import LSTM, BatchNormalization
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Nadam

    _, max_case_length, num_features = training_sequence.shape

    t0 = time.process_time()
    main_input = Input(shape=(max_case_length, num_features), name='main_input')
    # train a 2-layer LSTM with one shared layer
    l1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(
        main_input)  # the shared layer
    b1 = BatchNormalization()(l1)
    l2_1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(
        b1)  # the layer specialized in activity prediction
    b2_1 = BatchNormalization()(l2_1)

    # Output layer =====================================================================================================
    if n_lab > 1:
        output = \
            Dense(n_lab,
                  activation='softmax', kernel_initializer='glorot_uniform', name='output')(b2_1)
        loss = 'categorical_crossentropy'
    else:
        output = \
            Dense(1, kernel_initializer='glorot_uniform', name='output')(b2_1)
        loss = 'mae'

    # Compile model ====================================================================================================
    model = Model(inputs=[main_input], outputs=[output])

    opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

    model.compile(loss={'output': loss}, optimizer=opt)

    # Callbacks ========================================================================================================

    # Train / restart model ============================================================================================

    if fn_temp is not None and fn_temp.exists():
        print('\tloading previous model')
        model.load_weights(str(fn_temp))

    train_tax_model(model, training_sequence, validation_sequence, fn_temp)

    t1 = time.process_time()

    print(f'finished training in {t1 - t0} seconds')

    return model


def train_tax_model(model, training_sequence, validation_sequence, fn_temp):
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=42)

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0)

    callbacks = [early_stopping, lr_reducer]

    if fn_temp is not None:
        callbacks.append(ModelCheckpoint(str(fn_temp), monitor='val_loss',
                                         verbose=0, save_best_only=True, save_weights_only=False, mode='auto'))

    max_case_length = training_sequence.shape[1]

    model.fit(training_sequence, validation_data=validation_sequence, verbose=2,
              callbacks=callbacks, batch_size=max_case_length, epochs=500)

    return model
