import scipy as sp

def test_model(config, model, test_data):
    spearman_result = {}

    if config.transfer_learning:
        # TODO
        exit(2)

    else:
        print('Testing pre-train model')
        if config.enzyme == 'multi_task':
            enzymes = ['wt', 'esp', 'hf']
            for enzyme in enzymes:
                test_input = [test_data.enzymes_seq[enzyme].X, test_data.enzymes_seq[enzyme].X_biofeat]
                test_true_label = test_data.enzymes_seq[enzyme].y
                test_prediction = model.predict(test_input)
                spearman = sp.stats.spearmanr(test_true_label, test_prediction)[0]
                spearman_result[enzyme] = spearman
                print(f'Enzyme: {enzyme}, Spearman: {spearman}')
        else:
            if (config.model_type == 'gl_lstm') and (config.layer_num != 4):
                test_input = [test_data.enzymes_seq[config.enzyme].X]
            else:
                test_input = [test_data.enzymes_seq[config.enzyme].X, test_data.enzymes_seq[config.enzyme].X_biofeat]
            test_true_label = test_data.enzymes_seq[config.enzyme].y
            test_prediction = model.predict(test_input)
            spearman = sp.stats.spearmanr(test_true_label, test_prediction)[0]
            spearman_result[config.enzyme] = spearman
            print(f'Enzyme: {config.enzyme}, Spearman: {spearman}')

    return spearman_result




