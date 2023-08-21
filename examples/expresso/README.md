# Expresso
This repository shows an example of training a hifigan model [1,2] conditionned on one-hot speaker and style information as in the [Expresso paper](https://speechbot.github.io/expresso/) [3]. It also supports duration prediction module as in [4].

## Data
The data follows the code dataset format, where each line is expected to be a dictionary with the following keys: "audio", "hubert" (or "codes"), "spk", "style".

For example
```
{'audio': 'path/to/audio_1.wav', 'hubert': '17 17 296 296 296...', 'spk': 'speaker_03', 'style': 'style_16'}
{'audio': 'path/to/audio_2.wav', 'hubert': '79 79 79 487 288...', 'spk': 'speaker_01', 'style': 'style_05'}
...
```

## Training
To train the hifigan model, firstly you need to prepare a config file. An example of the config file can be found in the [config directory](config/expresso_config.json). Remember to specify the paths to the training and validation datasets as well as the number of units (num_embeddings).

Run the following command to train the model
```bash
python -m torch.distributed.launch --nproc_per_node $NUM_GPUS \
    -m examples.expresso.train \
	--checkpoint_path $OUTPUT_DIR \
	--config $CONFIG_FILE
```

## Inference
You can use `examples/expresso/inference.py` to generate synthesized speech with the model.

The following command generate audio from code input with random speaker and style
```bash
python -m examples.expresso.inference \
    --input_code_file $INPUT_CODE_FILE \
    --checkpoint_file $CHECKPOINT_FILE \
    --output_dir $OUTPUT_DIR \
    --random_speaker \
    --random_speaker_subset ex_S07 ex_S08 \
    --random_style \
    --random_style_subset read-default read-happy read-sad read-laughing \
    -n $NUM_FILES \
    --num-gpu $NUM_GPUS \
```

You can also use `textlesslib` to synthesize speech from hifigan model as described in the [Expresso repo](https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso).

## References
[1] [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646) \
[2] [Speech Resynthesis from Discrete Disentangled Self-Supervised Representations](https://arxiv.org/abs/2104.00355) \
[3] [Expresso: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis](https://arxiv.org/abs/2308.05725) \
[4] [Textless Speech-to-Speech Translation on Real Data](https://arxiv.org/abs/2112.08352)