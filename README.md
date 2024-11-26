# SovitsTokenizer

SpeechTokenizer: A low-bitrate audio tokenizer that converts speech into discrete tokens (as low as 25 tokens per second) while preserving semantic and prosodic richness. Leveraging the pre-trained SoVITs model from [GPT-SoVITs](https://github.com/RVC-Boss/GPT-SoVITS), it fine-tunes VQ-VAE layers for efficient audio compression and utilizes HuBERT’s robust semantic extraction. By combining HuBERT’s deep linguistic understanding with VQ-VAE’s detailed capture of prosodic and phonetic features, and decoupled text embedding from MTRE module, SpeechTokenizer produces compact yet highly expressive speech discrete units, which can also be used for voice conversion by decoding the tokens with a reference audio for speaker adaptation.

## Installation

```bash
git clone https://github.com/hon9kon9ize/sovits-tokenizer.git
cd sovits-tokenizer
pip install -e .
```

## Usage

```python
from sovits_tokenizer import SovitsTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator_weights = "pretrained_models/s2G2333k.pth" # download from https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v2final-pretrained
hubert_base_path = "pretrained_models/chinese-hubert-base" # download from https://huggingface.co/lj1995/GPT-SoVITS/tree/main/chinese-hubert-base

speech_tokenizer = SpeechTokenizer(generator_weights, hubert_base_path, device=device)

print(codes.shape) # (1, 1, 538) batch_size, codebook_size, seq_len
print(outputs.shape) # (688640,)
print("duration", outputs.shape[0] / 16000) # duration 21 
print("TBS", codes.shape[-1] / math.ceil(outputs.shape[0] / 34000)) # TBS 25.61904761904762

# Reconstruction and Voice Conversion
reference_audio = "path/to/reference_audio.wav"
recon_wav = speech_tokenizer.decode(codes, reference_audio)
```

## Example

Input audio:

![Original Audio](./original.mp3)

Reconstructed audio:

![Reconstructed Audio](./recon.mp3)


## Acknowledgment and Inspiration

This project is inspired by and builds upon the work of [GPT-SoVITs](https://github.com/RVC-Boss/GPT-SoVITS). We borrow key ideas and components from the original repository, such as leveraging VQ-VAE and HuBERT for speech representation and semantic extraction. The innovations in GPT-SoVITS have been instrumental in shaping the foundation of SpeechTokenizer, extending its capabilities to efficient tokenization and voice conversion using reference audio.

Special thanks to the original developers of GPT-SoVITS for their groundbreaking contributions to the field!