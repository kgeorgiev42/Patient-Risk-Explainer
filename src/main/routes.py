import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

from flask import render_template, request, current_app, session, jsonify, url_for

from src.main import bp
from src.model import ExplainerController, Predictor, Explainer, Evaluator
from src.model.modules.utils import clean_text, tokenize, filter_text, puncts, odd_chars, contraction_mapping

handler = RotatingFileHandler('application.log', maxBytes=10000, backupCount=3)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
logger.addHandler(handler)


@bp.route('/')
@bp.route('/home', methods=['GET', 'POST'])
def home():
    """Renders the home page."""
    if request.method == 'GET':
        return render_template(
            'index.html',
            year=datetime.now().year,
        )

    raw_letter = request.form['md-form-text']
    selected_model = request.form['model-select']
    selected_task = request.form['task-select']
    selected_threshold = request.form['threshold-select']

    if selected_threshold == 'Default':
        selected_threshold = 8 if selected_model == 'fasttext' else 10

    prep_letter = clean_text(raw_letter)
    prep_letter = tokenize(prep_letter)
    prep_letter = filter_text(prep_letter)

    #### 2. Download model files
    controller = ExplainerController(selected_model, selected_task, 'config/model_config.yml', 'config/data_config.yml')
    controller.download_model_files()
    ft_model, tokenizer, risk_grps = controller.load_model_files()

    #### 3. Initialize predictor and explainer
    predictor = Predictor(tokenizer, ft_model, risk_grps,
                          controller.model_config['explainers'][selected_task]['class_names'])
    explainer = Explainer(tokenizer, ft_model,
                          controller.model_config['explainers'][selected_task]['class_names'], predictor)

    #### 4. Extract predictions
    prep_letter = explainer.preprocess_set([prep_letter], puncts, odd_chars, contraction_mapping)
    letter_pred_dict = predictor.extract_set_preds([prep_letter])
    letter_df = predictor.preds_to_df(letter_pred_dict, raw_letter)

    #### 5. Evaluate with LIME and Sentence Ranking
    evaluator = Evaluator(explainer, predictor, letter_df, threshold=selected_threshold)
    lime_df = evaluator.lime_eval_predictions()
    lime_df = evaluator.lime_extract_explainer_stats(lime_df)
    full_df = evaluator.sr_eval_predictions(lime_df)
    full_df = evaluator.sr_extract_stats(full_df)
    top_positive = ""
    top_negative = ""
    top_positive_score = 0.0
    top_negative_score = 0.0

    for i in range(len(full_df.iloc[0].Explanations)):
        if full_df.iloc[0].Explanations[i][1] > 0:
            top_positive = full_df.iloc[0].Explanations[i][0]
            top_positive_score = full_df.iloc[0].Explanations[i][1]
            break

    for i in range(len(full_df.iloc[0].Explanations)):
        if full_df.iloc[0].Explanations[i][1] < 0:
            top_negative = full_df.iloc[0].Explanations[i][0]
            top_negative_score = full_df.iloc[0].Explanations[i][1]
            break

    logging.info('Explanation complete.. generating results.')

    session['risk_score'] = str(full_df.iloc[0].Risk)
    session['risk_per'] = str(round(full_df.iloc[0].Risk_per * 100, 2))
    session['risk_conf'] = str(round(float(full_df.iloc[0].Confidence) * 100, 2))
    session['m_exp_score'] = str(round(full_df.iloc[0].Mean_exp_score, 4))
    session['m_sent_r_score'] = str(round(full_df.iloc[0].Mean_Rank_Sent_Score, 4))
    session['t_pos_exp'] = str(top_positive)
    session['t_neg_exp'] = str(top_negative)
    session['t_pos_exp_s'] = str(round(top_positive_score, 4))
    session['t_neg_exp_s'] = str(round(top_negative_score, 4))
    session['t_pos_sent_exp_keys'] = [el[0] for el in full_df.iloc[0].S_Top_Pos_Ranks.items()]
    session['t_neg_sent_exp_keys'] = [el[0] for el in full_df.iloc[0].S_Top_Neg_Ranks.items()]
    session['t_pos_sent_exp_vals'] = [round(el[1], 4) for el in full_df.iloc[0].S_Top_Pos_Ranks.items()]
    session['t_neg_sent_exp_vals'] = [round(el[1], 4) for el in full_df.iloc[0].S_Top_Neg_Ranks.items()]
    session['explanations'] = json.dumps([(el[0], el[1]) for el in full_df.iloc[0].Explanations])
    session['risk_str'] = 'High risk' if full_df.iloc[0].Risk >= selected_threshold else 'Low risk'
    #session['raw_letter'] = full_df.iloc[0].Raw_Text

    return jsonify(list(session.items()))

@bp.route('/explain')
def explain():
    """Renders the home page."""
    return render_template(
        'explain.html',
        year=datetime.now().year,
    )

