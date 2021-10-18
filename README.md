# Speech Resynthesis from Discrete Disentangled Self-Supervised Representations
Implementation of the method described in the [Speech Resynthesis from Discrete Disentangled Self-Supervised Representations](https://arxiv.org/abs/2104.00355).

<p align="center"><img width="70%" src="img/fig.png" /></p>

__Abstract__: We propose using self-supervised discrete representations for the task of speech resynthesis. To generate disentangled representation, we separately extract low-bitrate representations for speech content, prosodic information, and speaker identity. This allows to synthesize speech in a controllable manner. We analyze various state-of-the-art, self-supervised representation learning methods and shed light on the advantages of each method while considering reconstruction quality and disentanglement properties. Specifically, we evaluate the F0 reconstruction, speaker identification performance (for both resynthesis and voice conversion), recordings' intelligibility, and overall quality using subjective human evaluation. Lastly, we demonstrate how these representations can be used for an ultra-lightweight speech codec. Using the obtained representations, we can get to a rate of 365 bits per second while providing better speech quality than the baseline methods.


## Quick Links
- [Samples](https://speechbot.github.io/resynthesis/index.html)
- [Setup](#setup)
- [Training](#training)
- [Inference](#inference)

## Setup

### Software
Requirements:
* Python >= 3.6
* PyTorch v1.8
* Install dependencies
    ```bash
    git clone https://github.com/facebookresearch/speech-resynthesis.git
    cd speech-resynthesis
    pip install -r requirements.txt
    ```
### Data

#### For LJSpeech:
1. Download LJSpeech dataset from [here](https://keithito.com/LJ-Speech-Dataset/) into ```data/LJSpeech-1.1``` folder.
2. Downsample audio from 22.05 kHz to 16 kHz and pad
   ```
   bash
   python ./scripts/preprocess.py \
   --srcdir data/LJSpeech-1.1/wavs \
   --outdir data/LJSpeech-1.1/wavs_16khz \
   --pad
   ```

#### For VCTK:
1. Download VCTK dataset from [here](https://datashare.ed.ac.uk/handle/10283/3443) into ```data/VCTK-Corpus``` folder.
2. Downsample audio from 48 kHz to 16 kHz, trim trailing silences and pad
   ```bash
   python ./scripts/preprocess.py \
   --srcdir data/VCTK-Corpus/wav48_silence_trimmed \
   --outdir data/VCTK-Corpus/wav16_silence_trimmed_padded \
   --pad --postfix mic2.flac
   ```

## Training

#### F0 Quantizer Model
To train F0 quantizer model, use the following command:
   ```bash
   python -m torch.distributed.launch --nproc_per_node 8 train_f0_vq.py \
   --checkpoint_path checkpoints/lj_f0_vq \
   --config configs/LJSpeech/f0_vqvae.json
   ```
Set ```<NUM_GPUS>``` to the number of availalbe GPUs on your machine.

#### Resynthesis Model
To train a resynthesis model, use the following command:
   ```bash
   python -m torch.distributed.launch --nproc_per_node <NUM_GPUS> train.py \
   --checkpoint_path checkpoints/lj_vqvae \
   --config configs/LJSpeech/vqvae256_lut.json
   ```

#### Supported Configurations
Currently, we support the following training schemes:

|Dataset|SSL Method|Dictionary Size|Config Path|
|------|---|---|---|
|LJSpeech|HuBERT|100|```configs/LJSpeech/hubert100_lut.json```|
|LJSpeech|CPC|100|```configs/LJSpeech/cpc100_lut.json```|
|LJSpeech|VQVAE|256|```configs/LJSpeech/vqvae256_lut.json```|
|VCTK|HuBERT|100|```configs/VCTK/hubert100_lut.json```|
|VCTK|CPC|100|```configs/VCTK/cpc100_lut.json```|
|VCTK|VQVAE|256|```configs/VCTK/vqvae256_lut.json```|

## Inference
To generate, simply run:
```bash
python inference.py \
--checkpoint_file checkpoints/vctk_cpc100 \
-n 10 \
--output_dir generations
```

To synthesize multiple speakers:
```bash
python inference.py \
--checkpoint_file checkpoints/vctk_cpc100 \
-n 10 \
--vc \
--input_code_file datasets/VCTK/cpc100/test.txt \
--output_dir generations_multispkr
```

You can also generate with codes from a different dataset:
```bash
python inference.py \
--checkpoint_file checkpoints/lj_cpc100 \
-n 10 \
--input_code_file datasets/VCTK/cpc100/test.txt \
--output_dir generations_vctk_to_lj
```

## Preprocessing New Datasets

#### CPC / HuBERT Coding
To quantize new datasets with CPC or HuBERT follow the instructions described in the [GSLM code](https://github.com/pytorch/fairseq/tree/master/examples/textless_nlp/gslm).

To parse CPC output:
```bash
python scripts/parse_cpc_codes.py \
--manifest cpc_output_file \
--wav-root wav_root_dir \
--outdir parsed_cpc
```

To parse HuBERT output:
```bash
python parse_hubert_codes.py \
--codes hubert_output_file \
--manifest hubert_tsv_file \
--outdir parsed_hubert 
```

#### VQVAE Coding
First, you will need to download [LibriLight](https://github.com/facebookresearch/libri-light) dataset and move it to ```data/LibriLight```.

For VQVAE, train a vqvae model using the following command:
```bash
python -m torch.distributed.launch --nproc_per_node <NUM_GPUS> train.py \
--checkpoint_path checkpoints/ll_vq \
--config configs/LibriLight/vqvae256.json
```

To extract VQVAE codes:
```bash
python infer_vqvae_codes.py \
--input_dir folder_with_wavs_to_code \
--output_dir vqvae_output_folder \
--checkpoint_file checkpoints/ll_vq
```

To parse VQVAE output:
```bash
 python parse_vqvae_codes.py \
 --manifest vqvae_output_file \
 --outdir parsed_vqvae
```

## License

You may find out more about the license [here](https://github.com/facebookresearch/speech-resynthesis/blob/main/LICENSE).

## Citation
```
@inproceedings{polyak21_interspeech,
  author={Adam Polyak and Yossi Adi and Jade Copet and 
          Eugene Kharitonov and Kushal Lakhotia and 
          Wei-Ning Hsu and Abdelrahman Mohamed and Emmanuel Dupoux},
  title={{Speech Resynthesis from Discrete Disentangled Self-Supervised Representations}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
}
``` 

## Acknowledgements
This implementation uses code from the following repos: [HiFi-GAN](https://github.com/jik876/hifi-gan) and [Jukebox](https://github.com/openai/jukebox), as described in our code.