import os
import argparse

from utils.functions import Storage

class ConfigRegression_r():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'misa': self.__MISA
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = 'E:\yanyi\code of paper\MISA-master\MISA-master\Dataset'
        tmp = {
            'mosi':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                }
            },
            'mosei':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'unaligned': {
                    #'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    #'train_samples': 16326,
                    'train_samples': 10620,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                }
            },
            'sims':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/unaligned_39.pkl'),
                    #'dataPath': os.path.join(root_dataset_dir, 'SIMS/unaligned_39.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55), # (text, audio, video)
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'train_samples': 1368,
                    #'num_classes': 3,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                }
            }
        }
        return tmp

    def __MISA(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
                'save_labels': False,
                'early_stop': 8,
                'update_epochs': 4,
                'rnncell':"lstm",
                "use_cmd_sim": True
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    "batch_size": 16,
                    "learning_rate": 0.0001,
                    "hidden_size": 128,
                    "dropout": 0.2,
                    "reverse_grad_weight": 0.8,
                    "diff_weight": 0.1,
                    "sim_weight": 0.3,
                    "sp_weight": 1.0,
                    "recon_weight": 1.0,
                    "grad_clip": 0.8,
                    "weight_decay": 0.0,
                    "transformers": "bert",
                    "pretrained": "bert-base-uncased"
                },
                'mosei':{
                    "batch_size": 64,
                    "learning_rate": 0.0001,
                    "hidden_size": 128,
                    "dropout": 0.5,
                    "reverse_grad_weight": 0.8,
                    "diff_weight": 0.3,
                    "sim_weight": 0.8,
                    "sp_weight": 1.0,
                    "recon_weight": 1.0,
                    "grad_clip": 0.8,
                    "weight_decay": 5e-5,
                    "transformers": "bert",
                    "pretrained": "bert-base-uncased"
                },
                'sims':{
                    "batch_size": 64,
                    "learning_rate": 0.0001,
                    "hidden_size": 128,
                    "dropout": 0.0,
                    "reverse_grad_weight": 0.5,
                    "diff_weight": 0.3,
                    "sim_weight": 1.0,
                    "sp_weight": 1.0,
                    "recon_weight": 0.8,
                    "grad_clip": 1.0,
                    "weight_decay": 5e-5,
                    "transformers": "bert",
                    "pretrained": "bert-base-chinese"
                },
                'simsv2': {
                    "batch_size": 16,
                    "learning_rate": 0.0001,
                    "hidden_size": 64,
                    "dropout": 0.5,
                    "reverse_grad_weight": 0.5,
                    "diff_weight": 0.5,
                    "sim_weight": 1.0,
                    "sp_weight": 0.0,
                    "recon_weight": 0.5,
                    "grad_clip": 0.8,
                    "weight_decay": 5e-5,
                    "transformers": "bert",
                    "pretrained": "bert-base-chinese"
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args