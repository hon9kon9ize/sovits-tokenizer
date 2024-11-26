import os
from typing import List
from .feature_extractor.hubert import HuBERT
from .module.models import SynthesizerTrn
from .utils import get_wav, get_spepc, HParams, DictToAttrRecursive
import numpy as np
import json
import torch


class SovitsTokenizer:
    """
    Speech audio tokenizer, which extracts latent codes from the speech audio by using the HuBERT model and the VQ-VAE model.

    Args:
        config_path (str): path to the config file
        model_path (str): path to the model file
        hubert_path (str): path to the hubert model file
        device (str): device to use (default: None)
        is_half (bool): whether to use half precision (default: False), which can reduce the memory usage and speed up the computation, but may lead to a slight decrease in the accuracy
    """

    def __init__(
        self, model_path, hubert_path, config_path=None, device=None, is_half=False
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        state_dict = torch.load(model_path, map_location="cpu")

        if config_path is not None:
            with open(config_path, "r") as f:
                data = f.read()
            config = json.loads(data)
            hps = HParams(**config)
        elif "config" in state_dict:
            hps = state_dict["config"]
            hps = DictToAttrRecursive(hps)
        else:
            raise ValueError(
                "config_path should be provided if the config is not saved in the model file"
            )

        hps.model.semantic_frame_rate = "25hz"

        self.hps = hps
        self.is_half = is_half
        self.hubert = HuBERT(hubert_path).to(device).eval()
        self.vqvae = (
            SynthesizerTrn(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                **hps.model,
            )
            .to(device)
            .eval()
        )

        if is_half:
            self.hubert = self.hubert.half()
            self.vqvae = self.vqvae.half()

        model_weights = (
            state_dict["model"]
            if "model" in state_dict
            else state_dict["weight"]  # backward compatibility with older checkpoints
        )

        self.vqvae.load_state_dict(model_weights, strict=False)

    @property
    def sampling_rate(self) -> int:
        return self.hps.data.sampling_rate

    def encode(self, audio_path) -> List[int]:
        wav16k = get_wav(audio_path, sr=16000, device=self.device)

        if self.is_half:
            wav16k = wav16k.half()

        ssl_content = self.hubert(wav16k).transpose(1, 2)
        codes = self.vqvae.extract_latent(ssl_content)

        return codes

    def decode(self, codes: List[int], refer_audio_path: str) -> np.ndarray:
        refer = get_spepc(self.hps, refer_audio_path, device=self.device)

        if self.is_half:
            refer = refer.half()

        outputs = self.vqvae.decode(codes, [refer])

        return outputs.squeeze().cpu().numpy()

    def __call__(self, audio_path) -> List[int]:
        return self.encode(audio_path)

    # class method `from_pretrained` is used to create a new instance of the class from the pretrained model
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        hubert_name: str = None,
        device: str = None,
        is_half: bool = False,
    ):
        # check if model_name is a folder
        if os.path.isdir(model_name):
            model_path = os.path.join(model_name, "model.pth")
            config_path = os.path.join(model_name, "config.json")
        else:
            raise ValueError("model_name should be a folder")

        with open(config_path, "r") as f:
            data = f.read()
        config = json.loads(data)

        if "hubert_model_name_or_path" in config and hubert_name is None:
            hubert_name = config["hubert_model_name_or_path"]

        if hubert_name is None:
            raise ValueError(
                "hubert_model_name_or_path should be provided in the config file if hubert_name is not provided"
            )

        return cls(model_path, hubert_name, config_path, device=device, is_half=is_half)
