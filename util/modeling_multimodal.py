# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, set_caching_enabled
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
root = "/Users/pvbastidas/repoTesis/datasets_tesis/vqa/dataset_daquar/"
from transformers import (
    # Preprocessing / Common
    AutoTokenizer, AutoFeatureExtractor,
    # Text & Image Models (Now, image transformers like ViTModel, DeiTModel, BEiT can also be loaded using AutoModel)
    AutoModel,            
    # Training / Evaluation
    TrainingArguments, Trainer,
    # Misc
    logging
)

# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet

from sklearn.metrics import accuracy_score, f1_score
#ini pvbm para aumentar
"""
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
"""
##ini para q no muestre warnings de la version beta de v2
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
##fin para q no muestre warnings de la version beta de v2

from torchvision.transforms.functional import InterpolationMode
#fin pvbm para aumentar
class MultimodalCollator:
    def __init__(self, tokenizer:AutoTokenizer, preprocessor:AutoFeatureExtractor):
        # Instance variables
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor

        
    def tokenize_text(self, texts: List[str]):
        encoded_text = self.tokenizer(
            text=texts,
            padding='longest',
            max_length=24,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text['input_ids'].squeeze(),
            "token_type_ids": encoded_text['token_type_ids'].squeeze(),
            "attention_mask": encoded_text['attention_mask'].squeeze(),
        }

    def preprocess_images(self, images: List[str]):
        #ini pvbm esto es para aumentar datos quitar si no sirve
        size = (self.preprocessor.size["height"], self.preprocessor.size["width"])
        #print('esto es el tamanooooo:'+str(size))
        """original
        train_transforms = Compose(
            [
                #RandomResizedCrop(size),
                RandomHorizontalFlip(),
                
            ]
        )
        """
        train_transforms = v2.Compose(
            [
                #v2.Resize(size, interpolation=InterpolationMode.BICUBIC),
                # CenterCrop(image_size),
                #v2.RandomCrop(size, pad_if_needed=True, padding_mode="edge"),
                #v2.ColorJitter(hue=0.1),
                v2.RandomHorizontalFlip(),
                #v2.RandomRotation(15, interpolation=InterpolationMode.BILINEAR, fill=128),
                v2.RandomAutocontrast(p=0.3),
                v2.RandomEqualize(p=0.3), 
                
            ]
        )


        
        #fin pvbm esto es para aumentar datos quitar si no sirve
        processed_images = self.preprocessor(
            
            #images=[Image.open(os.path.join(root,"images", image_id + ".png")).convert('RGB') for image_id in images], #esto es el que vale original
            #images=[Image.open(os.path.join("..", "dataset", "images", image_id + ".png")).convert('RGB') for image_id in images], esto era el original pvbm
            images=[train_transforms(Image.open(os.path.join(root, "images", image_id + ".png")).convert('RGB')) for image_id in images],#pvbm para aumentar
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }
            
    def __call__(self, raw_batch_dict):
        return {
            **self.tokenize_text(
                raw_batch_dict['question']
                if isinstance(raw_batch_dict, dict) else
                [i['question'] for i in raw_batch_dict]
            ),
            **self.preprocess_images(
                raw_batch_dict['image_id']
                if isinstance(raw_batch_dict, dict) else
                [i['image_id'] for i in raw_batch_dict]
            ),
            'labels': torch.tensor(
                raw_batch_dict['label']
                if isinstance(raw_batch_dict, dict) else
                [i['label'] for i in raw_batch_dict],
                dtype=torch.int64
            ),
        }