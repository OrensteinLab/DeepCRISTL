def train_model(config, DataHandler, model, callback_list, verbose=2, final_models=False):
    if verbose > 0:
        print('Start training')
    train_input, y_train = [DataHandler['X_train'], DataHandler['dg_train']], DataHandler['y_train']
    valid_input, y_val = [DataHandler['X_valid'], DataHandler['dg_valid']], DataHandler['y_valid']


    history = model.fit(train_input,
                        y_train,
                        batch_size=config.batch_size,
                        epochs=config.epochs,
                        verbose=verbose,
                        validation_data=(valid_input, y_val),
                        shuffle=True,
                        callbacks=callback_list,
                        )
    if not final_models:
        for key in history.history.keys():
            print(key, history.history[key])

    return history

