# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import argparse
import json
import os
import random
import time
from multiprocessing import Manager, Pool

import librosa
import numpy as np
import torch
from scipy.io.wavfile import write

from utils import AttrDict
from examples.expresso.dataset import InferenceCodeDataset
from examples.expresso.models import MultiSpkrMultiStyleCodeGenerator
from tqdm import tqdm

h = None
device = None

from inference import scan_checkpoint, load_checkpoint, generate


def scan_generator_checkpoint(checkpoint_file):
    if os.path.isdir(checkpoint_file):
        cp_g = scan_checkpoint(checkpoint_file, "g_")
    else:
        cp_g = checkpoint_file

    assert os.path.exists(cp_g) and os.path.isfile(
        cp_g
    ), f"Didn't find checkpoints for {cp_g}"

    return cp_g


def load_config(checkpoint_file):
    ckpt_dir = os.path.dirname(checkpoint_file)
    config_file = os.path.join(ckpt_dir, "config.json")

    with open(config_file) as f:
        json_config = json.loads(f.read())

    config = AttrDict(json_config)

    return config


def load_generator(checkpoint_file, cuda_idx):
    generator = MultiSpkrMultiStyleCodeGenerator(h).to(cuda_idx)

    cp_g = scan_generator_checkpoint(checkpoint_file)
    state_dict_g = load_checkpoint(cp_g)
    generator.load_state_dict(state_dict_g["generator"])

    generator.eval()
    generator.remove_weight_norm()

    return generator


def load_vocoder_meta(vocoder_ckpt):
    if vocoder_ckpt is None:
        return None, None
    ckpt_dir = os.path.dirname(vocoder_ckpt)

    spkr_file = os.path.join(ckpt_dir, "speakers.txt")
    speakers = None
    if os.path.exists(spkr_file):
        with open(spkr_file) as f:
            speakers = [line.strip() for line in f]
        print(f"Loaded {len(speakers)} speakers. First few speakers: {speakers[:10]}")

    style_file = os.path.join(ckpt_dir, "styles.txt")
    styles = None
    if os.path.exists(style_file):
        with open(style_file) as f:
            styles = [line.strip() for line in f]
        print(f"Loaded {len(styles)} styles. First few styles: {styles[:10]}")

    return speakers, styles


def init_worker(queue, arguments):
    import logging

    logging.getLogger().handlers = []

    global generator
    global idx
    global device
    global args

    args = arguments
    idx = queue.get()
    device = idx

    generator = load_generator(args.checkpoint_file, idx)


@torch.inference_mode()
def inference(item_index):
    code, gt_audio, filename, fname_out_name = dataset[item_index]
    code = {k: torch.from_numpy(v).to(device).unsqueeze(0) for k, v in code.items()}

    new_code = dict(code)

    if args.dur_prediction:
        assert (
            generator.dur_predictor is not None
        ), "Model doesn't support duration prediction!"
        new_code["code"] = torch.unique_consecutive(new_code["code"]).unsqueeze(0)
        new_code["dur_prediction"] = True

    audio, rtf = generate(h, generator, new_code)
    output_file = os.path.join(args.output_dir, fname_out_name + "_gen.wav")
    audio = librosa.util.normalize(audio.astype(np.float32))
    write(output_file, h.sampling_rate, audio)

    if gt_audio is not None and not args.not_write_gt:
        output_file = os.path.join(args.output_dir, fname_out_name + "_gt.wav")
        gt_audio = librosa.util.normalize(gt_audio.squeeze().astype(np.float32))
        write(output_file, h.sampling_rate, gt_audio)


def main():
    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_code_file",
        required=True,
        help="Input code file with the code dataset format "
        "(i.e. each line is a dictionary represents a file)",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory to store the wavs"
    )
    parser.add_argument(
        "--checkpoint_file",
        required=True,
        help="Path to the hifigan generator checkpoint",
    )
    parser.add_argument(
        "--forced_speaker", type=str, help="Force a specific speaker name"
    )
    parser.add_argument(
        "--random_speaker",
        action="store_true",
        help="Randomly sample a speaker for each file",
    )
    parser.add_argument(
        "--random_speaker_subset", nargs="+", help="Subset of speaker to sample on"
    )
    parser.add_argument("--forced_style", type=str, help="Force a specific style name")
    parser.add_argument(
        "--random_style",
        action="store_true",
        help="Randomly sample style for each file",
    )
    parser.add_argument(
        "--random_style_subset", nargs="+", help="Subset of style to sample on"
    )
    parser.add_argument(
        "--parts",
        action="store_true",
        help="Write output filename with 3 last parts in the path",
    )
    parser.add_argument(
        "--not_write_gt",
        action="store_true",
        help="Don't write ground truth audio along with generated audio",
    )
    parser.add_argument(
        "-n", type=int, default=-1, help="Restrict to a number of files"
    )
    parser.add_argument("--num-gpu", type=int, default=2, help="Number of gpus")
    parser.add_argument(
        "--dur-prediction",
        action="store_true",
        help="Predict duration (if model supports duration prediction)",
    )
    parser.add_argument("--debug", action="store_true", help="Run debug with 1 gpu")
    args = parser.parse_args()

    random.seed(1234)

    global h
    global dataset

    scan_generator_checkpoint(args.checkpoint_file)
    h = load_config(args.checkpoint_file)

    speakers, styles = load_vocoder_meta(args.checkpoint_file)
    dataset = InferenceCodeDataset(
        input_code_file=args.input_code_file,
        name_parts=args.parts,
        sampling_rate=h.sampling_rate,
        multispkr=h.get("multispkr", None),
        speakers=speakers,
        forced_speaker=args.forced_speaker,
        random_speaker=args.random_speaker,
        random_speaker_subset=args.random_speaker_subset,
        multistyle=h.get("multistyle", None),
        styles=styles,
        forced_style=args.forced_style,
        random_style=args.random_style,
        random_style_subset=args.random_style_subset,
    )

    ids_data = list(range(len(dataset)))
    if not args.debug:
        random.shuffle(ids_data)

    n_data = len(dataset) if args.n < 0 else min(args.n, len(dataset))
    ids_data = ids_data[:n_data]

    print(f"Output directory: '{args.output_dir}'")
    if len(ids_data) > 0:
        os.makedirs(args.output_dir, exist_ok=True)

    num_gpu = 1 if args.debug else args.num_gpu
    print(f"Generating {len(ids_data)} sequences with {num_gpu} gpus...")
    stime = time.time()
    if args.debug:
        import queue

        ids = list(range(1))
        idQueue = queue.Queue()
        for i in ids:
            idQueue.put(i)
        init_worker(idQueue, args)

        for i in tqdm(ids_data):
            inference(i)
    else:
        ids = list(range(8))
        manager = Manager()
        idQueue = manager.Queue()
        for i in ids:
            idQueue.put(i)

        with Pool(args.num_gpu, init_worker, (idQueue, args)) as pool:
            for _ in tqdm(pool.imap(inference, ids_data), total=n_data):
                continue
    print(f"...done in {time.time()-stime:.2f} seconds!")


if __name__ == "__main__":
    main()
