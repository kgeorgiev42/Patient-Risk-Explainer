import itertools
import os
import re
from collections import OrderedDict
from statistics import mean

import nltk
from scipy import stats
from tqdm import tqdm

nltk.download('stopwords')

import lime
from lime import lime_text, submodular_pick


class BaseExplainer():
    '''
    Base Explainer module, initialized with the model file, tokenizer and class names.
    '''

    def __init__(self, tokenizer, clf, class_names):
        self.tokenizer = tokenizer
        self.clf = clf
        self.class_names = class_names

    def remove_stopwords(self, words):
        """
        Function to remove stopwords from the question text
        """
        stop_words = set(nltk.corpus.stopwords.words("english"))
        return [word for word in words if word not in stop_words]

    def remove_punctuation(self, text):
        """
        Function to remove punctuation from the question text
        """
        return re.sub(r'[^\w\s]', '', text)

    def lemmatize_text(self, words):
        """
        Function to lemmatize the question text
        """
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in words]

    def stem_text(self, words):
        """
        Function to stem the question text
        """
        ps = nltk.stem.PorterStemmer()
        return [ps.stem(word) for word in words]

    def clean_numbers(self, x):
        x = re.sub('[0-9]{5,}', ' ##### ', x)
        x = re.sub('[0-9]{4}', ' #### ', x)
        x = re.sub('[0-9]{3}', ' ### ', x)
        x = re.sub('[0-9]{2}', ' ## ', x)
        return x

    def punct_add_space(self, x, puncts):
        x = str(x)
        for punct in puncts:
            x = x.replace(punct, f' {punct} ')
        return x

    def odd_add_space(self, x, odd_chars):
        x = str(x)
        for odd in odd_chars:
            x = x.replace(odd, f' {odd} ')
        return x

    def clean_contractions(self, text, mapping):
        specials = ["’", "‘", "´", "`"]
        for s in specials:
            text = text.replace(s, "'")
        text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
        return text

    def preprocess_set(self, data, puncts, odd_chars, contraction_mapping):
        letters = []
        for i in tqdm(range(len(data))):
            sent_str = ' '.join(data[i])
            sent_str = self.clean_numbers(sent_str)
            sent_str = self.punct_add_space(sent_str, puncts)
            sent_str = self.odd_add_space(sent_str, odd_chars)
            sent_str = self.clean_contractions(sent_str, contraction_mapping)
            sent_str_tokens = nltk.tokenize.word_tokenize(sent_str)
            # sent_str_tokens = self.remove_stopwords(sent_str_tokens)
            # sent_str_tokens = self.lemmatize_text(sent_str_tokens)
            sent_str_post = ' '.join(sent_str_tokens)
            # sent_str_post = sent_str_post.replace('. ', '.\n')
            letters.append(sent_str_post)

        return letters

    def seq_prediction(self, classifier, tokenizer):
        '''
        Prediction method to be inherited by each model.
        '''
        pass

    def seq_explain(self, sentences, top_labels=1, sample_size=5000, num_features=20):
        '''
        Explanation method to be inherited by each model.
        '''
        pass

    def save_as_html(self, exp, path):
        '''
        Save the explanation to an HTML file so it's easy to view.
        You can also get it to other formats: as_list(), as_map(), etc.
        See https://lime-ml.readthedocs.io/en/latest/lime.html#lime.explanation.Explanation
        '''
        try:
            os.makedirs('/'.join(path.split('/')[:-1]))
        except:
            pass
        exp.save_to_file(path)
        print('HTML explanation file generated.')


class Explainer(BaseExplainer):
    '''
    1D-CNN explanation module for LIME. Includes perturbation analysis, sentence splitting and
    ranking.
    sample_size - number of samples to perturb
    num_features - number of words/sentences to score
    '''

    def __init__(self, tokenizer, clf, class_names, predictor):
        super().__init__(tokenizer, clf, class_names)
        self.predictor = predictor

    def seq_explain(self, sentences, top_labels=1, sample_size=5000, num_features=20):
        '''
        Runs the LIME explainer using N-grams. Modified version includes sentence perturbation.
        Returns the mapping of useful words/clauses.
        '''
        # pre_letter = '\n'.join(sentences)
        explainer = lime.lime_text.LimeTextExplainer(
            bow=False,
            class_names=self.predictor.class_names
        )
        exp = explainer.explain_instance(
            sentences,
            self.predictor.seq_predict,
            top_labels=top_labels,
            num_features=num_features,
            num_samples=sample_size
        )
        return exp, exp.as_list()

    def seq_explain_global(self, letters, sample_size=2, num_features=6, num_exps_desired=2):
        explainer = lime.lime_text.LimeTextExplainer(
            bow=False,
            class_names=self.predictor.class_names
        )
        sp_obj = submodular_pick.SubmodularPick(explainer,
                                                letters,
                                                self.predictor.seq_predict,
                                                sample_size=sample_size,
                                                num_features=num_features,
                                                num_exps_desired=num_exps_desired)
        return sp_obj.sp_explanations

    def split_sentences(self, letter, min_len=30):
        '''
        Word tokenizing function for the ranking module.
        '''
        # separate sentences in new lines and strip extra whitespace
        letter_pre = self.strip_formatting(letter)
        letter_pre = re.sub(' +', ' ', letter_pre)
        # print(letter_pre)
        nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = nltk_tokenizer.tokenize(letter_pre)
        filtered_sentences = []

        ### Filter out first and last paragraphs
        for sent in sentences[1:-1]:
            if sent[0].isalpha() and not sent[-2].isdigit() and len(sent) >= min_len:
                filtered_sentences.append(sent)

        return filtered_sentences

    def rank_sentences(self, sentences, seq_map, true_cls=None, top=5):
        '''
        Sentence ranking module.
        Retrieves the word weights from the LIME explainer and normalizes them using z-score [-1, 1].
        Afterwards, the word dictionary is used to (naively) sum up the scores for each sentence, by
        the occuring words and their weights.
        '''
        prob_scores = self.predictor.seq_predict(sentences)
        prob_mean = mean([prob_scores[i][1] for i in range(len(prob_scores))])

        print('Probability of risk (positive):', prob_mean)
        pred_cls = self.class_names[1] if prob_mean > self.predictor.risk_grps[-1] else self.class_names[0]
        print('Predicted class:', pred_cls)
        if true_cls is not None:
            print('True class:', true_cls)

        idx_risk = self.predictor.get_risk_grp(prob_mean)
        print('Predicted risk group:', idx_risk)
        print('Perturbation sample size:', self.sample_size)
        print('Number of words scored:', self.num_features)
        print()
        reverse_word_map = dict(map(reversed, self.tokenizer.word_index.items()))
        stop_words = set(nltk.corpus.stopwords.words('english'))
        seq_values = list(seq_map.values())[0]
        ranks = {}
        words = []
        scores = []
        # print(seq_values)
        for i in range(len(seq_values)):
            if seq_values[i][0] in reverse_word_map.keys():
                if reverse_word_map[seq_values[i][0]] not in stop_words and "unkwordz" not in reverse_word_map[
                    seq_values[i][0]]:
                    if reverse_word_map[seq_values[i][0]].isdigit() == False and len(
                            reverse_word_map[seq_values[i][0]]) > 3:
                        words.append(reverse_word_map[seq_values[i][0]])
                        scores.append(seq_values[i][1])

        # set weights as normalized z-score
        weights = stats.zscore(scores)
        word_dict = dict(zip(words, weights))
        print('LIME word vectors:')
        for k, v in word_dict.items():
            print('{}: {}'.format(k, v))

        print()

        sent_list = sentences.split('\n')
        for sent in sent_list:
            sent_score = 0.0
            word_tokens = nltk.word_tokenize(sent)
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

        print('Top {} Positive sentences:'.format(top))
        for k, v in sorted_pos_sent.items():
            print('---------------')
            print('{}\n Sentence Score ({})'.format(k, v))
            print('---------------')
            print()

        print()
        print('Top {} Negative sentences:'.format(top))
        for k, v in sorted_neg_sent.items():
            print('---------------')
            print('{}\n Sentence Score ({})'.format(k, v))
            print('---------------')
            print()

        return sorted_neg_sent, sorted_pos_sent
