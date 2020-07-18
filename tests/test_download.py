import logging
import os
import unittest
from src.model import ExplainerController

def test_download_fasttext():
    selected_model = 'fasttext'
    selected_task = 'rohr_task'

    logging.info('Testing fasttext model download..')
    try:
        os.makedirs('data/pickles/')
    except:
        pass
    controller = ExplainerController(selected_model, selected_task, 'config/model_config.yml', 'config/data_config.yml')
    controller.download_model_files()
    ft_model, tokenizer, risk_grps = controller.load_model_files()
    assert(len(risk_grps) > 0 and len(tokenizer.word_index) > 0)
    assert(ft_model.layers is not None)

def test_download_clstm():
    selected_model = '1d_cnn'
    selected_task = 'rohr_task'

    logging.info('Testing 1d-conv-lstm model download..')
    try:
        os.makedirs('data/pickles/')
    except:
        pass
    controller = ExplainerController(selected_model, selected_task, 'config/model_config.yml', 'config/data_config.yml')
    controller.download_model_files()
    cnn_model, tokenizer, risk_grps = controller.load_model_files()
    assert(len(risk_grps) > 0 and len(tokenizer.word_index) > 0)
    assert(cnn_model.layers is not None)
