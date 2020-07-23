import itertools
from collections import OrderedDict

import nltk

nltk.download('stopwords')

import pandas as pd
from nltk import word_tokenize
from scipy import stats
from tqdm import tqdm


class Evaluator():
    """
    Evaluator class for running the statistics on the model explanations.

    Params:
        explainer(modules.Explainer): the explanation module
        predictor(modules.Predictor): the prediction module
        pred_df(pandas.df): DataFrame containing the model predictions/explanations
    """
    def __init__(self, explainer, predictor, pred_df):
        self.explainer = explainer
        self.predictor = predictor
        self.pred_df = pred_df

    def lime_eval_predictions(self, sample_size=5000, num_features=20):
        """
        Run the evaluation on the LIME module.
        :param sample_size(int): number of samples to perturb per instance
        :param num_features(int): number of features to score per sample
        :return: pd.DataFrame(the concatenated DataFrame with the original containing the evaluation results)
        """
        eval_df = pd.DataFrame()
        for idx, row in tqdm(self.pred_df.iterrows()):
            exp, exp_map = self.explainer.seq_explain(row.Text, sample_size, num_features)
            print('Saving explanation..')
            eval_df = eval_df.append({
                'Sample_size': sample_size,
                'N_features': num_features,
                'Explanations': list(exp_map),
                'Exp_seq_map': list(exp.as_map().values())[0]}, ignore_index=True)

        # eval_df = eval_df.drop(labels='Unnamed: 0', axis=1)
        return pd.concat([self.pred_df, eval_df], axis=1)

    def lime_count_pos(self, exp_list):
        """
        Count the amount of positives in the list of explanations.
        :param exp_list(list): a sample list from the DataFrame
        :return: float(Positive ratio of samples)
        """
        ctr_pos = 0
        for exp in exp_list:
            if exp[1] >= 0:
                ctr_pos += 1

        return float(ctr_pos / len(exp_list))

    def lime_count_neg(self, exp_list):
        """
        Count the amount of negatives in the list of explanations.
        :param exp_list(list): a sample list from the DataFrame
        :return: float(Negative ratio of samples)
        """
        ctr_neg = 0
        for exp in exp_list:
            if exp[1] < 0:
                ctr_neg += 1

        return float(ctr_neg / len(exp_list))

    def lime_sum_pos(self, exp_list):
        """
        Sum all of the positive explanation scores in the list.
        :param exp_list(list): a sample list from the DataFrame
        :return: float(Sum of positive scores)
        """
        s_pos = 0.0
        for exp in exp_list:
            if exp[1] >= 0:
                s_pos += exp[1]

        return s_pos

    def lime_sum_neg(self, exp_list):
        """
        Sum all of the negative explanation scores in the list.
        :param exp_list(list): a sample list from the DataFrame
        :return: float(Sum of negative scores)
        """
        s_neg = 0.0
        for exp in exp_list:
            if exp[1] < 0:
                s_neg += exp[1]

        return s_neg

    def lime_extract_explainer_stats(self, df):
        """
        Estimate the LIME explainer metrics with lambda functions and append them to the DataFrame.
        :param df(pd.DataFrame): original prediction DataFrame
        :return: df(The processed DataFrame)
        """
        df["Letter_Length"] = df["Text"].apply(lambda x: len(x))
        df["Pos_ratio"] = df["Explanations"].apply(lambda x: self.lime_count_pos(x))
        df["Neg_ratio"] = df["Explanations"].apply(lambda x: self.lime_count_neg(x))
        df["Sum_pos"] = df["Explanations"].apply(lambda x: self.lime_sum_pos(x))
        df["Sum_neg"] = df["Explanations"].apply(lambda x: self.lime_sum_neg(x))
        df["Top_exp"] = df["Explanations"].apply(lambda x: x[0][1])
        df["Top_sent_length"] = df["Explanations"].apply(lambda x: len(x[0][0]))
        df["Mean_exp_score"] = df["Sum_pos"] + df["Sum_neg"]

        return df

    def sr_eval_predictions(self, df, top=5):
        """
        Run the evaluation on the Sentence Rankings.
        :param df(pd.DataFrame): DataFrame containing the predictions.
        :param top(int): Number of sentences to rank.
        :return: df(The processed DataFrame with top positive/negative sentence scores)
        """
        reverse_word_map = dict(map(reversed, self.predictor.tokenizer.word_index.items()))
        stop_words = set(nltk.corpus.stopwords.words('english'))
        word_dicts = []
        ranks_list = []
        top_pos_ranks = []
        top_neg_ranks = []

        ### Get a list of sentences from the letter text
        df["Text_Split"] = df["Text"].apply(lambda x: ('.\n'.join(x.split(". "))).split('\n'))

        for idx, row in tqdm(df.iterrows()):
            ranks = {}
            words = []
            scores = []
            sentence_scores = row.Exp_seq_map
            for i in range(len(sentence_scores)):
                if sentence_scores[i][0] in reverse_word_map.keys():
                    if reverse_word_map[sentence_scores[i][0]] not in stop_words and "unkwordz" not in reverse_word_map[
                        sentence_scores[i][0]]:
                        if reverse_word_map[sentence_scores[i][0]].isdigit() == False and len(
                                reverse_word_map[sentence_scores[i][0]]) > 3:
                            words.append(reverse_word_map[sentence_scores[i][0]])
                            scores.append(sentence_scores[i][1])

            # set weights as normalized z-score
            weights = stats.zscore(scores)
            word_dict = dict(zip(words, weights))
            sent_list = row.Text_Split
            for sent in sent_list:
                sent_score = 0.0
                word_tokens = word_tokenize(sent)
                filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
                for word in filtered_sentence:
                    if word.lower() in list(word_dict.keys()):
                        sent_score += word_dict[word.lower()]
                ranks[sent] = sent_score

            sorted_ranks = {k: v for k, v in sorted(ranks.items(), key=lambda item: item[1])}
            sorted_ranks_r = {k: v for k, v in sorted(ranks.items(), key=lambda item: item[1], reverse=True)}
            neg_sent = dict(itertools.islice(sorted_ranks.items(), top))
            pos_sent = dict(itertools.islice(sorted_ranks_r.items(), top))
            sorted_neg_sent = OrderedDict(neg_sent)
            sorted_pos_sent = OrderedDict(pos_sent)

            top_pos_ranks.append(sorted_pos_sent)
            top_neg_ranks.append(sorted_neg_sent)
            ranks_list.append(ranks)
            word_dicts.append(word_dict)

        df['S_Word_Dict'] = word_dicts
        df['S_Ranks'] = ranks_list
        df['S_Top_Pos_Ranks'] = top_pos_ranks
        df['S_Top_Neg_Ranks'] = top_neg_ranks

        return df

    def sr_get_length(self, pos_dict):
        """
        Get the total sum of sentence lengths per example (positive or negative).
        :param pos_dict(dict): the dictionary of ranked sentences
        :return: s_lengths(int)
        """
        s_lengths = 0
        for k in pos_dict.keys():
            s_lengths += len(k)

        return s_lengths

    def sr_get_score_sum(self, pos_dict):
        """
        Get the total sum of sentence scores per example (positive or negative).
        :param pos_dict(dict): the dictionary of ranked sentences
        :return: s_sum(int)
        """
        s_sum = 0.0
        for v in pos_dict.values():
            s_sum += v

        return s_sum

    def sr_extract_stats(self, eval_df):
        """
        Get the sentence ranking metrics using lambda functions and append them to the DataFrame.
        :param eval_df(pd.DataFrame): DataFrame containing the predictions + the LIME explainer metrics
        :return: eval_df(pd.DataFrame: the full evaluation DataFrame)
        """
        eval_df['Top_Pos_Sent_length'] = eval_df['S_Top_Pos_Ranks'].apply(lambda x: self.sr_get_length(x))
        eval_df['Top_Neg_Sent_length'] = eval_df['S_Top_Neg_Ranks'].apply(lambda x: self.sr_get_length(x))
        eval_df['Top_Pos_Sent_Score_Sum'] = eval_df['S_Top_Pos_Ranks'].apply(lambda x: self.sr_get_score_sum(x))
        eval_df['Top_Neg_Sent_Score_Sum'] = eval_df['S_Top_Neg_Ranks'].apply(lambda x: self.sr_get_score_sum(x))
        eval_df['Mean_Rank_Sent_Score'] = eval_df['S_Ranks'].apply(lambda x: self.sr_get_score_sum(x))
        eval_df['Mean_Rank_Word_Score'] = eval_df['S_Word_Dict'].apply(lambda x: self.sr_get_score_sum(x))

        return eval_df
