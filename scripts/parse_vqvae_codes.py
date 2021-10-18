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


def parse_manifest(manifest):
    audio_files = []

    with open(manifest) as info:
        for line in info.readlines():
            if line[0] == '{':
                sample = eval(line.strip())
                audio_files += [Path(sample["audio"])]
            else:
                audio_files += [Path(line.strip())]

    return audio_files


def split(args, samples):
    if args.ref_train is not None:
        train_split = parse_manifest(args.ref_train)
        train_split = [x.name for x in train_split]
        val_split = parse_manifest(args.ref_val)
        val_split = [x.name for x in val_split]
        test_split = parse_manifest(args.ref_test)
        test_split = [x.name for x in test_split]
        tt = []
        cv = []
        tr = []

        # parse
        for sample in samples:
            name = Path(sample['audio']).name
            if name in val_split:
                cv += [sample]
            elif name in test_split:
                tt += [sample]
            else:
                tr += [sample]
                assert name in train_split
    else:
        # split
        N = len(samples)
        random.shuffle(samples)
        tt = samples[: int(N * args.tt)]
        cv = samples[int(N * args.tt): int(N * args.tt + N * args.cv)]
        tr = samples[int(N * args.tt + N * args.cv):]

    return tr, cv, tt


def save(outdir, tr, cv, tt):
    # save
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outdir / f'train.txt', 'w') as f:
        f.write('\n'.join([str(x) for x in tr]))
    with open(outdir / f'val.txt', 'w') as f:
        f.write('\n'.join([str(x) for x in cv]))
    with open(outdir / f'test.txt', 'w') as f:
        f.write('\n'.join([str(x) for x in tt]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--cv', type=float, default=0.05)
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
