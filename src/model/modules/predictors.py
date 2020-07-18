import numpy as np
import pandas as pd
from src.model.modules.explainers import BaseExplainer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


class Predictor(BaseExplainer):
    '''
    1D-CNN Prediction class for a Keras-based classifier with tokenization.
    '''

    def __init__(self, tokenizer, clf, risk_grps, class_names, threshold):
        super().__init__(tokenizer, clf, class_names)
        self.MAX_SEQ_LENGTH = 6489
        self.risk_grps = risk_grps
        self.threshold = threshold

    def extract_set_preds(self, data, labels=None):
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
        val_df = pd.DataFrame(data=[pred_dict.keys(), pred_dict.values()]).T
        if true_label is not None:
            values_df = pd.DataFrame(val_df[1].to_list(), columns=['Confidence', 'Risk', 'Pred_label', 'True_label', 'Risk_per'])
        values_df = pd.DataFrame(val_df[1].to_list(), columns=['Confidence', 'Risk', 'Pred_label', 'Risk_per'])
        values_df['Confidence'] = [''.join(map(str, l)) for l in values_df['Confidence']]
        values_df.insert(0, 'Text', val_df[0])
        values_df.insert(0, 'Raw_Text', raw_letter)
        return values_df

    def get_risk_grp(self, prob):
        for i in range(len(self.risk_grps) - 1):
            if prob >= self.risk_grps[i] and prob <= self.risk_grps[i + 1]:
                return i + 1

        return len(self.risk_grps)

    def seq_predict_cls(self, letter):
        '''
        Sequence prediction function. Performs Keras-based word tokenization and padding.
        '''
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
        # classes_probs = [float(i) for i in classes_probs]

        # print(classes_probs)

        return np.array([classes_probs], dtype='float32')

    def seq_predict(self, letter, local=True):
        '''
        Sequence prediction function. Performs Keras-based word tokenization and padding.
        '''
        # self.tokenizer.fit_on_texts(letter)
        if local:
            seq = self.tokenizer.texts_to_sequences(letter)
        else:
            seq = self.tokenizer.texts_to_sequences([letter])
        text_data = pad_sequences(seq, maxlen=self.MAX_SEQ_LENGTH)
        probs = self.clf.predict(text_data)
        new_probs = []
        for i in range(len(probs)):
            new_probs.append([1 - probs[i][0], probs[i][0]])

        # print('Mean positive:', mean([new_probs[i][0] for i in range(len(new_probs))]))

        # print(new_probs)
        return np.array(new_probs)
