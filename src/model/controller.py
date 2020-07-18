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
        if os.path.isdir(self.model_config['explainers'][self.task][self.model_type]['path']):
            logging.info('Model already exists.. Skipping download')
            return

        if not os.path.exists(self.model_config['explainers'][self.task][self.model_type]['path']):
            os.makedirs(os.path.dirname(self.model_config['explainers'][self.task][self.model_type]['path']))
            os.makedirs(os.path.dirname(self.model_config['explainers'][self.task][self.model_type]['path'] + 'assets/'))
            os.makedirs(os.path.dirname(self.model_config['explainers'][self.task][self.model_type]['path'] + 'variables/'))

        for object in bucket.objects.filter(Prefix=self.model_config['explainers'][self.task][self.model_type]['path'] + 'variables/'):
            bucket.download_file(object.key, object.key)

        bucket.download_file(self.model_config['explainers'][self.task][self.model_type]['path'] + 'saved_model.pb',
                             self.model_config['explainers'][self.task][self.model_type]['path'] + 'saved_model.pb')

        bucket.download_file(self.model_config['explainers'][self.task][self.model_type]['risk_path'],
                             self.model_config['explainers'][self.task][self.model_type]['risk_path'])

        bucket.download_file(self.model_config['explainers'][self.task]['tokenizer_path'],
                             self.model_config['explainers'][self.task]['tokenizer_path'])

        logging.info('Model download successful.')

    def load_model_files(self):
        if self.model_type not in ['fasttext', '1d_cnn']:
            logging.error('Model type not supported.')
            return
        if not os.path.isdir(self.model_config['explainers'][self.task][self.model_type]['path']):
            logging.info('Model not downloaded yet.. call download first.')
            return

        model = models.load_model(self.model_config['explainers'][self.task][self.model_type]['path'])
        with open(self.model_config['explainers'][self.task]['tokenizer_path'], 'rb') as handle:
            tokenizer = pickle.load(handle)

        with open(self.model_config['explainers'][self.task][self.model_type]['risk_path'], 'rb') as handle:
            risk_grps = pickle.load(handle)

        return model, tokenizer, risk_grps
