# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import argparse
import glob
import json
import os
import random
import sys
import time
from multiprocessing import Manager, Pool
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.io.wavfile import write

from dataset import CodeDataset, parse_manifest, mel_spectrogram, \
    MAX_WAV_VALUE
from utils import AttrDict
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


def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location='cpu')
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def generate(h, generator, code):
    start = time.time()
    y_g_hat = generator(**code)
    if type(y_g_hat) is tuple:
        y_g_hat = y_g_hat[0]
    rtf = (time.time() - start) / (y_g_hat.shape[-1] / h.sampling_rate)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    return audio, rtf


def init_worker(queue, arguments):
    import logging
    logging.getLogger().handlers = []

    global generator
    global f0_stats
    global spkrs_emb
    global dataset
    global spkr_dataset
    global idx
    global device
    global a
    global h
    global spkrs

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


    if a.code_file is not None:
        dataset = [x.strip().split('|') for x in open(a.code_file).readlines()]

        def parse_code(c):
            c = [int(v) for v in c.split(" ")]
            return [torch.LongTensor(c).numpy()]

        dataset = [(parse_code(x[1]), None, x[0], None) for x in dataset]
    else:
        file_list = parse_manifest(a.input_code_file)
        dataset = CodeDataset(file_list, -1, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                              h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device,
                              f0=h.get('f0', None), multispkr=h.get('multispkr', None),
                              f0_stats=h.get('f0_stats', None), f0_normalize=h.get('f0_normalize', False),
                              f0_feats=h.get('f0_feats', False), f0_median=h.get('f0_median', False),
                              f0_interp=h.get('f0_interp', False), vqvae=h.get('code_vq_params', False),
                              pad=a.pad)

    if a.unseen_f0:
        dataset.f0_stats = torch.load(a.unseen_f0)

    os.makedirs(a.output_dir, exist_ok=True)

    if h.get('multispkr', None):
        spkrs = random.sample(range(len(dataset.id_to_spkr)), k=min(5, len(dataset.id_to_spkr)))

    if a.f0_stats and h.get('f0', None) is not None:
        f0_stats = torch.load(a.f0_stats)

    generator.eval()
    generator.remove_weight_norm()

    # fix seed
    seed = 52 + idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def inference(item_index):
    code, gt_audio, filename, _ = dataset[item_index]
    code = {k: torch.from_numpy(v).to(device).unsqueeze(0) for k, v in code.items()}

    if a.parts:
        parts = Path(filename).parts
        fname_out_name = '_'.join(parts[-3:])[:-4]
    else:
        fname_out_name = Path(filename).stem

    if h.get('f0_vq_params', None) or h.get('f0_quantizer', None):
        to_remove = gt_audio.shape[-1] % (16 * 80)
        assert to_remove % h['code_hop_size'] == 0

        if to_remove != 0:
            to_remove_code = to_remove // h['code_hop_size']
            to_remove_f0 = to_remove // 80

            gt_audio = gt_audio[:-to_remove]
            code['code'] = code['code'][..., :-to_remove_code]
            code['f0'] = code['f0'][..., :-to_remove_f0]

    new_code = dict(code)
    if 'f0' in new_code:
        del new_code['f0']
        new_code['f0'] = code['f0']

    audio, rtf = generate(h, generator, new_code)
    output_file = os.path.join(a.output_dir, fname_out_name + '_gen.wav')
    audio = librosa.util.normalize(audio.astype(np.float32))
    write(output_file, h.sampling_rate, audio)

    if h.get('multispkr', None) and a.vc:
        if a.random_speakers:
            local_spkrs = random.sample(range(len(dataset.id_to_spkr)), k=min(5, len(dataset.id_to_spkr)))
        else:
            local_spkrs = spkrs

        for spkr_i, k in enumerate(local_spkrs):
            code['spkr'].fill_(k)

            if a.f0_stats and h.get('f0', None) is not None and not h.get('f0_normalize', False):
                spkr = k
                f0 = code['f0'].clone()

                ii = (f0 != 0)
                mean_, std_ = f0[ii].mean(), f0[ii].std()
                if spkr not in f0_stats:
                    new_mean_, new_std_ = f0_stats['f0_mean'], f0_stats['f0_std']
                else:
                    new_mean_, new_std_ = f0_stats[spkr]['f0_mean'], f0_stats[spkr]['f0_std']

                f0[ii] -= mean_
                f0[ii] /= std_
                f0[ii] *= new_std_
                f0[ii] += new_mean_
                code['f0'] = f0

            if h.get('f0_feats', False):
                f0_stats_ = torch.load(h["f0_stats"])
                if k not in f0_stats_:
                    mean = f0_stats_['f0_mean']
                    std = f0_stats_['f0_std']
                else:
                    mean = f0_stats_[k]['f0_mean']
                    std = f0_stats_[k]['f0_std']
                code['f0_stats'] = torch.FloatTensor([mean, std]).view(1, -1).to(device)

            audio, rtf = generate(h, generator, code)

            output_file = os.path.join(a.output_dir, fname_out_name + f'_{k}_gen.wav')
            audio = librosa.util.normalize(audio.astype(np.float32))
            write(output_file, h.sampling_rate, audio)

    if gt_audio is not None:
        output_file = os.path.join(a.output_dir, fname_out_name + '_gt.wav')
        gt_audio = librosa.util.normalize(gt_audio.squeeze().numpy().astype(np.float32))
        write(output_file, h.sampling_rate, gt_audio)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--code_file', default=None)
    parser.add_argument('--input_code_file', default='./datasets/LJSpeech/cpc100/test.txt')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--f0-stats', type=Path)
    parser.add_argument('--vc', action='store_true')
    parser.add_argument('--random-speakers', action='store_true')
    parser.add_argument('--pad', default=None, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--parts', action='store_true')
    parser.add_argument('--unseen-f0', type=Path)
    parser.add_argument('-n', type=int, default=10)
    a = parser.parse_args()

    seed = 52
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ids = list(range(8))
    manager = Manager()
    idQueue = manager.Queue()
    for i in ids:
        idQueue.put(i)

    if os.path.isdir(a.checkpoint_file):
        config_file = os.path.join(a.checkpoint_file, 'config.json')
    else:
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    if not os.path.isfile(cp_g) or not os.path.exists(cp_g):
        print(f"Didn't find checkpoints for {cp_g}")
        return

    if a.code_file is not None:
        dataset = [x.strip().split('|') for x in open(a.code_file).readlines()]

        def parse_code(c):
            c = [int(v) for v in c.split(" ")]
            return [torch.LongTensor(c).numpy()]

        dataset = [(parse_code(x[1]), None, x[0], None) for x in dataset]
    else:
        file_list = parse_manifest(a.input_code_file)
        dataset = CodeDataset(file_list, -1, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                              h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss, device=device,
                              f0=h.get('f0', None), multispkr=h.get('multispkr', None),
                              f0_stats=h.get('f0_stats', None), f0_normalize=h.get('f0_normalize', False),
                              f0_feats=h.get('f0_feats', False), f0_median=h.get('f0_median', False),
                              f0_interp=h.get('f0_interp', False), vqvae=h.get('code_vq_params', False),
                              pad=a.pad)

    if a.debug:
        ids = list(range(1))
        import queue
        idQueue = queue.Queue()
        for i in ids:
            idQueue.put(i)
        init_worker(idQueue, a)

        for i in range(0, len(dataset)):
            inference(i)
            bar = progbar(i, len(dataset))
            message = f'{bar} {i}/{len(dataset)} '
            stream(message)
            if a.n != -1 and i > a.n:
                break
    else:
        idx = list(range(len(dataset)))
        random.shuffle(idx)
        with Pool(8, init_worker, (idQueue, a)) as pool:
            for i, _ in enumerate(pool.imap(inference, idx), 1):
                bar = progbar(i, len(idx))
                message = f'{bar} {i}/{len(idx)} '
                stream(message)
                if a.n != -1 and i > a.n:
                    break


if __name__ == '__main__':
    main()
