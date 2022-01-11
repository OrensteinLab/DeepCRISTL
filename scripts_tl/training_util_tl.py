def train_model(config, DataHandler, model, callback_list, verbose=2):
    if verbose > 0:
        print('Start training')
    train_input, y_train = [DataHandler['X_train'], DataHandler['X_biofeat_train']], DataHandler['y_train']
    valid_input, y_val = [DataHandler['X_valid'], DataHandler['X_biofeat_valid']], DataHandler['y_valid']

    if config.flanks:
        train_input += [DataHandler['up_train'], DataHandler['down_train']]
        valid_input += [DataHandler['up_valid'], DataHandler['down_valid']]

    if config.new_features:
        train_input += [DataHandler['new_features_train']]
        valid_input += [DataHandler['new_features_valid']]


    history = model.fit(train_input,
                        y_train,
                        batch_size=config.batch_size,
                        epochs=config.epochs,
                        verbose=verbose,
                        validation_data=(valid_input, y_val),
                        shuffle=True,
                        callbacks=callback_list,
                        )
    return history

