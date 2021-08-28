# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from scripts.parse_cpc_codes import split, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--codes', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--min-dur', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ref-train', type=Path)
    parser.add_argument('--ref-val', type=Path)
    parser.add_argument('--ref-test', type=Path)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.manifest) as f:
        fnames = [l.strip() for l in f.readlines()]
    wav_root = Path(fnames[0])
    fnames = fnames[1:]

    with open(args.codes) as f:
        codes = [l.strip() for l in f.readlines()]

    # parse
    samples = []
    for fname_dur, code in tqdm(zip(fnames, codes)):
        sample = {}
        fname, dur = fname_dur.split('\t')

        sample['audio'] = str(wav_root / f'{fname}')
        sample['hubert'] = ' '.join(code.split(' '))
        sample['duration'] = int(dur) / 16000

        if args.min_dur and sample['duration'] < args.min_dur:
            continue

        samples += [sample]

    tr, cv, tt = split(args, samples)
    save(args.outdir, tr, cv, tt)


if __name__ == '__main__':
    main()
