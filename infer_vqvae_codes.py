# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import random
import sys
from multiprocessing import Manager, Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from utils import AttrDict
from inference import load_checkpoint, scan_checkpoint
from models import CodeGenerator

h = None
device = None


def stream(message):
    sys.stdout.write(f"\r{message}")


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def init_worker(queue, arguments):
    import logging
    logging.getLogger().handlers = []

    global encoder
    global vq
    global dataset
    global idx
    global device
    global a
    global h

    a = arguments
    idx = queue.get()
    device = idx

    if os.path.isdir(a.checkpoint_file):
        config_file = os.path.join(a.checkpoint_file, 'config.json')
    else:
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = CodeGenerator(h).to(idx)
    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    state_dict_g = load_checkpoint(cp_g)
    generator.load_state_dict(state_dict_g['generator'])

    encoder = generator.code_encoder
    encoder.eval()

    vq = generator.code_vq
    vq.eval()

    # fix seed
    seed = 52
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def inference(path):
    # total_rtf = 0.0
    audio, sr = sf.read(path)
    audio = torch.from_numpy(audio).view(1, 1, -1)
    audio = audio.to(device).float()
    h = encoder(audio)
    code, _, _, _ = vq(h)
    code = code[0].cpu().squeeze()
    code = ",".join([str(x.item()) for x in code])

    return str(path), code


def main():
    print('Initializing VQVAE Extraction Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('-n', type=int, default=-1)
    parser.add_argument('--ext', type=str, default="wav")
    a = parser.parse_args()

    ids = list(range(8))
    manager = Manager()
    idQueue = manager.Queue()
    for i in ids:
        idQueue.put(i)

    files = a.input_dir.glob(f'**/*{a.ext}')
    files = list(files)
    lines = []

    if a.gpus > 1:
        with Pool(a.gpus, init_worker, (idQueue, a)) as pool:
            for i, l in enumerate(pool.imap(inference, files), 1):
                bar = progbar(i, len(files))
                message = f'{bar} {i}/{len(files)} '
                stream(message)
                lines += [l]
                if a.n != -1 and i > a.n:
                    break
    else:
        ids = list(range(1))
        import queue
        idQueue = queue.Queue()
        for i in ids:
            idQueue.put(i)
        init_worker(idQueue, a)

        for i, p in enumerate(files):
            l = inference(p)
            lines += [l]
            bar = progbar(i, len(files))
            message = f'{bar} {i}/{len(files)} '
            stream(message)
            if a.n != -1 and i > a.n:
                break

    a.output_dir.mkdir(exist_ok=True)
    with open(a.output_dir / 'vqvae_output.txt', 'w') as f:
        f.write("\n".join("\t".join(l) for l in lines))


if __name__ == '__main__':
    main()
