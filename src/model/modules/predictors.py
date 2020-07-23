import numpy as np
import pandas as pd
from src.model.modules.explainers import BaseExplainer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


class Predictor(BaseExplainer):
    """
    Base Predictor class for extracting predictions from Keras text classifiers.

    Params:
        tokenizer(list): the Keras tokenizer created from the respective dataset.
        clf(tf.Model): the tf.Keras model object
        class_names(list): list of positive/negative class labels
        risk_grps(list): pre-estimated risk decision boundaries
        threshold(int): the decision threshold for the positive/negative class [1-10]
    """

    def __init__(self, tokenizer, clf, risk_grps, class_names, threshold):
        super().__init__(tokenizer, clf, class_names)
        self.MAX_SEQ_LENGTH = 6489
        self.risk_grps = risk_grps
        self.threshold = int(threshold)

    def extract_set_preds(self, data, labels=None):
        """
        Extract the predictions/labels from the set of letters.
        :param data(list): list of preprocessed letters
        :param labels(list): list of ground truth labels
        :return: letter_dict(dict: contains the probability scores, risk scores and labels)
        """
        letter_dict = {}
        letters = []
        for i in range(len(data)):
            sent_str = ' '.join(data[i])
            letters.append(sent_str)

        seq = self.tokenizer.texts_to_sequences(letters)
        text_data = pad_sequences(seq, maxlen=self.MAX_SEQ_LENGTH)
        probs = self.clf.predict(text_data)
        for i in tqdm(range(len(probs))):
            risk_score = self.get_risk_grp(probs[i])
            pred_label = self.class_names[1] if risk_score >= self.threshold else self.class_names[0]
            if labels is not None:
                true_label = labels[i]
                letter_dict[letters[i]] = (probs[i], risk_score, pred_label, self.class_names[true_label], self.risk_grps[risk_score - 1])
            letter_dict[letters[i]] = (probs[i], risk_score, pred_label, self.risk_grps[risk_score - 1])

        return letter_dict

    def preds_to_df(self, pred_dict, raw_letter, true_label=None):
        """
        Converts the prediction dictionary to a DataFrame.
        :param pred_dict(dict): dictionary of the predictions
        :param raw_letter(str): the raw letter text for reference
        :param true_label(str): the name of the ground truth label
        :return: values_df(pd.DataFrame): the processed prediction DataFrame
        """
        val_df = pd.DataFrame(data=[pred_dict.keys(), pred_dict.values()]).T
        if true_label is not None:
            values_df = pd.DataFrame(val_df[1].to_list(), columns=['Confidence', 'Risk', 'Pred_label', 'True_label', 'Risk_per'])
        values_df = pd.DataFrame(val_df[1].to_list(), columns=['Confidence', 'Risk', 'Pred_label', 'Risk_per'])
        values_df['Confidence'] = [''.join(map(str, l)) for l in values_df['Confidence']]
        values_df.insert(0, 'Text', val_df[0])
        values_df.insert(0, 'Raw_Text', raw_letter)
        return values_df

    def get_risk_grp(self, prob):
        """
        Returns the risk score based on the model probability score.
        :param prob(float): probability of the positive class
        :return: int(The output risk score based on the pre-defined thresholds)
        """
        for i in range(len(self.risk_grps) - 1):
            if prob >= self.risk_grps[i] and prob <= self.risk_grps[i + 1]:
                return i + 1

        return len(self.risk_grps)

    def seq_predict_cls(self, letter):
        """
        Sequence prediction function (returns class labels). Performs Keras-based word tokenization and padding.
        :param letter(list): the preprocessed letter/letters
        :return: np.array(Array of classes for each sample)
        """
        seq = self.tokenizer.texts_to_sequences(letter)
        text_data = pad_sequences(seq, maxlen=self.MAX_SEQ_LENGTH)
        probs = self.clf.predict(text_data)

        classes_pred = []
        for prob in probs:
            if prob >= self.risk_grps[-1]:
                classes_pred.append(1)
            else:
                classes_pred.append(0)

        classes_probs = [classes_pred.count(0) / len(classes_pred), classes_pred.count(1) / len(classes_pred)]

        return np.array([classes_probs], dtype='float32')

    def seq_predict(self, letter, local=True):
        """
        Sequence prediction function (returns class probabilities). Performs Keras-based word tokenization and padding.
        :param letter(list): the preprocessed letter/letters
        :return: np.array(Array of probabilities for each sample)
        """
        if local:
            seq = self.tokenizer.texts_to_sequences(letter)
        else:
            seq = self.tokenizer.texts_to_sequences([letter])
        text_data = pad_sequences(seq, maxlen=self.MAX_SEQ_LENGTH)
        probs = self.clf.predict(text_data)
        new_probs = []
        for i in range(len(probs)):
            new_probs.append([1 - probs[i][0], probs[i][0]])

        return np.array(new_probs)
