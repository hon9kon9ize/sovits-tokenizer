import os
from transformers import logging as tf_logging

tf_logging.set_verbosity_error()

import logging

logging.getLogger("numba").setLevel(logging.WARNING)

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

import torch.nn as nn


class HuBERT(nn.Module):
    def __init__(self, base_path: str = None):
        super().__init__()
        self.model = HubertModel.from_pretrained(base_path, local_files_only=True)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            base_path, local_files_only=True
        )

    def forward(self, x):
        input_values = (
            self.feature_extractor(x, return_tensors="pt", sampling_rate=16000)
            .input_values.to(x.device)
            .to(x.dtype)
        )
        feats = self.model(input_values)["last_hidden_state"]
        return feats
