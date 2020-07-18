import logging
import os
import unittest
from src.model import ExplainerController, Predictor, Explainer, Evaluator
from src.model.modules.utils import clean_text, tokenize, filter_text, puncts, odd_chars, contraction_mapping


def test_low_threshold():
    with open('../samples/sample_letter.txt', 'r') as f:
        raw_letter = f.read()

    assert(len(raw_letter) > 0)
    prep_letter = clean_text(raw_letter)
    prep_letter = tokenize(prep_letter)
    prep_letter = filter_text(prep_letter)

    selected_model = 'fasttext'
    selected_task = 'rod_task'
    selected_threshold = 5
    logging.info('Testing risk of death task, threshold=5..')
    try:
        os.makedirs('data/pickles/')
    except:
        pass
    controller = ExplainerController(selected_model, selected_task, '../config/model_config.yml', '../config/data_config.yml')
    controller.download_model_files()
    ft_model, tokenizer, risk_grps = controller.load_model_files()
    predictor = Predictor(tokenizer, ft_model, risk_grps,
                          controller.model_config['explainers'][selected_task]['class_names'],
                          threshold=selected_threshold)
    explainer = Explainer(tokenizer, ft_model,
                          controller.model_config['explainers'][selected_task]['class_names'], predictor,
                          threshold=selected_threshold)
    prep_letter = explainer.preprocess_set([prep_letter], puncts, odd_chars, contraction_mapping)
    letter_pred_dict = predictor.extract_set_preds([prep_letter])
    letter_df = predictor.preds_to_df(letter_pred_dict, raw_letter)

    evaluator = Evaluator(explainer, predictor, letter_df)
    lime_df = evaluator.lime_eval_predictions()
    lime_df = evaluator.lime_extract_explainer_stats(lime_df)
    full_df = evaluator.sr_eval_predictions(lime_df)
    full_df = evaluator.sr_extract_stats(full_df)

    assert(full_df.iloc[0].Pred_label == 'Likely To Die' and full_df.iloc[0].Risk > selected_threshold)


def test_high_threshold():
    with open('../samples/sample_letter.txt', 'r') as f:
        raw_letter = f.read()

    assert (len(raw_letter) > 0)
    prep_letter = clean_text(raw_letter)
    prep_letter = tokenize(prep_letter)
    prep_letter = filter_text(prep_letter)

    selected_model = 'fasttext'
    selected_task = 'rod_task'
    selected_threshold = 10
    logging.info('Testing risk of death task, threshold=10..')
    try:
        os.makedirs('data/pickles/')
    except:
        pass
    controller = ExplainerController(selected_model, selected_task, '../config/model_config.yml',
                                     '../config/data_config.yml')
    controller.download_model_files()
    ft_model, tokenizer, risk_grps = controller.load_model_files()
    predictor = Predictor(tokenizer, ft_model, risk_grps,
                          controller.model_config['explainers'][selected_task]['class_names'],
                          threshold=selected_threshold)
    explainer = Explainer(tokenizer, ft_model,
                          controller.model_config['explainers'][selected_task]['class_names'], predictor,
                          threshold=selected_threshold)
    prep_letter = explainer.preprocess_set([prep_letter], puncts, odd_chars, contraction_mapping)
    letter_pred_dict = predictor.extract_set_preds([prep_letter])
    letter_df = predictor.preds_to_df(letter_pred_dict, raw_letter)

    evaluator = Evaluator(explainer, predictor, letter_df)
    lime_df = evaluator.lime_eval_predictions()
    lime_df = evaluator.lime_extract_explainer_stats(lime_df)
    full_df = evaluator.sr_eval_predictions(lime_df)
    full_df = evaluator.sr_extract_stats(full_df)

    assert (full_df.iloc[0].Pred_label == 'Not Likely To Die' and full_df.iloc[0].Risk <= selected_threshold)