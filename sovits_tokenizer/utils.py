import os
import sys
import json
import argparse
import traceback
import numpy as np
import ffmpeg
import torch
import glob
import librosa
import logging
from time import time as ttime
import shutil

logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logger = logging

from module.mel_processing import spectrogram_torch

zero_wav = np.zeros(
    int(16000 * 0.3),
    dtype=np.float32,
)


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "quantizer" not in k
            # print("load", k)
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except:
            traceback.print_exc()
            print(
                "error, %s is not in the checkpoint" % k
            )  # shape不对也会，比如text_embedding当cleaner修改时
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    print("load ")
    logger.info(
        "Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration)
    )
    return model, optimizer, learning_rate, iteration


def my_save(fea, path):  #####fix issue: torch.save doesn't support chinese path
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s.pth" % (ttime())
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir, name))


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    # torch.save(
    my_save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.ERROR)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_hparams(init=True, stage=1):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/s2.json",
        help="JSON file for configuration",
    )
    parser.add_argument(
        "-p", "--pretrain", type=str, required=False, default=None, help="pretrain dir"
    )
    parser.add_argument(
        "-rs",
        "--resume_step",
        type=int,
        required=False,
        default=None,
        help="resume step",
    )
    # parser.add_argument('-e', '--exp_dir', type=str, required=False,default=None,help='experiment directory')
    # parser.add_argument('-g', '--pretrained_s2G', type=str, required=False,default=None,help='pretrained sovits gererator weights')
    # parser.add_argument('-d', '--pretrained_s2D', type=str, required=False,default=None,help='pretrained sovits discriminator weights')

    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.pretrain = args.pretrain
    hparams.resume_step = args.resume_step
    # hparams.data.exp_dir = args.exp_dir
    if stage == 1:
        model_dir = hparams.s1_ckpt_dir
    else:
        model_dir = hparams.s2_ckpt_dir
    config_save_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(config_save_path, "w") as f:
        f.write(data)
    return hparams


def load_wav_to_torch(full_path):
    data, sampling_rate = librosa.load(full_path, sr=None)
    return torch.FloatTensor(data), sampling_rate


def get_wav(file, sr=16000, device=None):
    wav, sr = librosa.load(file, sr=sr)

    # convert to mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    wav = torch.from_numpy(wav)
    zero_wav_torch = torch.from_numpy(zero_wav)

    if device is not None:
        wav = wav.to(device)
        zero_wav_torch = zero_wav_torch.to(device)
    wav = torch.cat([wav, zero_wav_torch])

    return wav


def clean_path(path_str: str):
    if path_str.endswith(("\\", "/")):
        return clean_path(path_str[0:-1])
    path_str = path_str.replace("/", os.sep).replace("\\", os.sep)
    return (
        path_str.strip(" ").strip("'").strip("\n").strip('"').strip(" ").strip("\u202a")
    )


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
        if os.path.exists(file) == False:
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError("音频加载失败")

    return np.frombuffer(out, np.float32).flatten()


def get_spepc(hps, filename, device=None):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )

    if device is not None:
        spec = spec.to(device)

    return spec
