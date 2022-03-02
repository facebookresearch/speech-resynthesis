# Unit-based HiFi-GAN Vocoder with Duration Prediction

We provide implementation for the unit-based HiFi-GAN vocoder with a duration prediction module used in the direct speech-to-speech translation models in [1, 2].

## Training
```
# an example of training with HuBERT units

python -m torch.distributed.launch --nproc_per_node <NUM_GPUS> \
    -m examples.speech_to_speech_translation.train \
    --checkpoint_path checkpoints/lj_hubert100_dur1.0 \
    --config examples/speech_to_speech_translation/configs/hubert100_dw1.0.json
```

## Inference
To generate with duration prediction, simply run:
```
python -m examples.speech_to_speech_translation.inference \
    --checkpoint_file checkpoints/lj_hubert100_dur1.0 \
    -n 10 \
    --output_dir generations \
    --num-gpu <NUM_GPUS> \
    --input_code_file ./datasets/LJSpeech/hubert100/val.txt \
    --dur-prediction
```

#### fairseq
We also provide an implementation in [fairseq](https://github.com/pytorch/fairseq/tree/main/fairseq) for inference. See "Convert unit sequences to waveform" in the [example](https://github.com/pytorch/fairseq/tree/main/examples/speech_to_speech#inference).


## References
[1] [Direct speech-to-speech translation with discrete units](https://arxiv.org/abs/2107.05604) \
[2] [Textless Speech-to-Speech Translation on Real Data](https://arxiv.org/abs/2112.08352)