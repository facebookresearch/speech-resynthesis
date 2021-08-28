# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
from pathlib import Path

import torchaudio
from tqdm import tqdm

from scripts.parse_cpc_codes import split, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--cv', type=float, default=0.1)
    parser.add_argument('--tt', type=float, default=0.05)
    parser.add_argument('--min-dur', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ref-train', type=Path, default=None)
    parser.add_argument('--ref-val', type=Path, default=None)
    parser.add_argument('--ref-test', type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.manifest) as f:
        lines = [l.strip() for l in f.readlines()]

    # parse
    samples = []
    for l in tqdm(lines):
        sample = {}
        fname, code = l.split('\t')

        sample['audio'] = str(fname)
        sample['vqvae256'] = ' '.join(code.split(','))

        waveform, sample_rate = torchaudio.load(fname)
        sample['duration'] = waveform.shape[1] / sample_rate

        if args.min_dur and sample['duration'] < args.min_dur:
            continue
        samples += [sample]

    tr, cv, tt = split(args, samples)
    save(args.outdir, tr, cv, tt)


if __name__ == '__main__':
    main()
