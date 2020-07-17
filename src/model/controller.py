import logging
import os
import pickle

import boto3
import numpy as np
import yaml
from tensorflow.keras import models

AWS_BUCKET_NAME = 'prme-main'
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

s3_resource = boto3.resource('s3')
bucket = s3_resource.Bucket(AWS_BUCKET_NAME)


class ExplainerController():
    """
    XAI module controller template.

    """

    def __init__(self, model_type, task, model_config, data_config):
        self.model_type = model_type
        self.task = task
        self.model_config = self._load_config(model_config)
        self.data_config = self._load_config(data_config)

    def _load_config(self, config_path):
        with open(config_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return data

    def download_model_files(self):
        if self.model_type not in ['fasttext', '1d_cnn']:
            logging.error('Model type not supported.')
            return
        if os.path.isdir(self.model_config['explainers'][self.task][self.model_type]['download_path']):
            logging.info('Model already exists.. Skipping download')
            return

        for object in bucket.objects.filter(Prefix=self.model_config['explainers'][self.task][self.model_type]['path']):
            if not os.path.exists(os.path.dirname(object.key)):
                os.makedirs(os.path.dirname(object.key))
            bucket.download_file(object.key, object.key)

        bucket.download_file(self.model_config['explainers'][self.task][self.model_type]['risk_path'],
                             self.model_config['explainers'][self.task][self.model_type]['risk_path'])

        bucket.download_file(self.model_config['explainers'][self.task]['tokenizer_path'],
                             self.model_config['explainers'][self.task]['tokenizer_path'])

        logging.info('Model download successful.')

    def download_mimic_data(self):
        if os.path.isdir(self.data_config['data']['mimic']['path']):
            logging.info('Data already downloaded.. Skipping download')
            return

        bucket.download_file(os.path.join(self.data_config['data']['mimic']['path'], 'dc_train_sent.npy'),
                             os.path.join(self.data_config['data']['mimic']['download_path'], 'dc_train_sent.npy'))

        bucket.download_file(os.path.join(self.data_config['data']['mimic']['path'], 'dc_train_sent_lab.npy'),
                             os.path.join(self.data_config['data']['mimic']['download_path'], 'dc_train_sent_lab.npy'))

        bucket.download_file(os.path.join(self.data_config['data']['mimic']['path'], 'dc_val_sent.npy'),
                             os.path.join(self.data_config['data']['mimic']['download_path'], 'dc_val_sent.npy'))

        bucket.download_file(os.path.join(self.data_config['data']['mimic']['path'], 'dc_val_sent.npy'),
                             os.path.join(self.data_config['data']['mimic']['download_path'], 'dc_val_sent_lab.npy'))

        logging.info('Data download successful.')

    def load_model_files(self):
        if self.model_type not in ['fasttext', '1d_cnn']:
            logging.error('Model type not supported.')
            return
        if not os.path.isdir(self.model_config['explainers'][self.task][self.model_type]['download_path']):
            logging.info('Model not downloaded yet.. call download first.')
            return

        model = models.load_model(self.model_config['explainers'][self.task][self.model_type]['download_path'])
        with open(self.model_config['explainers'][self.task]['tokenizer_local_path'], 'rb') as handle:
            tokenizer = pickle.load(handle)

        with open(self.model_config['explainers'][self.task][self.model_type]['risk_local_path'], 'rb') as handle:
            risk_grps = pickle.load(handle)

        return model, tokenizer, risk_grps

    def load_mimic(self):
        if not os.path.isdir(self.data_config['data']['mimic']['path']):
            logging.info('Data not downloaded yet.. call download first.')
            return

        dc_train = np.load(os.path.join(self.data_config['data']['mimic']['download_path'], 'dc_train_sent.npy'),
                           allow_pickle=True)
        dc_train_labels = np.load(
            os.path.join(self.data_config['data']['mimic']['download_path'], 'dc_train_sent_lab.npy'))
        dc_val = np.load(os.path.join(self.data_config['data']['mimic']['download_path'], 'dc_val_sent.npy'),
                         allow_pickle=True)
        dc_val_labels = np.load(os.path.join(self.data_config['data']['mimic']['download_path'], 'dc_val_sent_lab.npy'))

        return dc_train, dc_train_labels, dc_val, dc_val_labels
