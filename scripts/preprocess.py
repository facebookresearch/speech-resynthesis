# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import resampy
import soundfile as sf
import librosa
from tqdm import tqdm


def pad_data(p, out_dir, trim=False, pad=False):
    data, sr = sf.read(p)
    if sr != 16000:
        data = resampy.resample(data, sr, 16000)
        sr = 16000

    if trim:
        data, _ = librosa.effects.trim(data, 20)

    if pad:
        if data.shape[0] % 1280 != 0:
            data = np.pad(data, (0, 1280 - data.shape[0] % 1280), mode='constant',
                          constant_values=0)
        assert data.shape[0] % 1280 == 0

    outpath = out_dir / p.name
    outpath.parent.mkdir(exist_ok=True, parents=True)
    sf.write(outpath, data, sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcdir', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--trim', action='store_true')
    parser.add_argument('--pad', action='store_true')
    parser.add_argument('--postfix', type=str, default='wav')
    args = parser.parse_args()

    files = list(Path(args.srcdir).glob(f'**/*{args.postfix}'))
    out_dir = Path(args.outdir)

    pad_data_ = partial(pad_data, out_dir=out_dir, trim=args.trim, pad=args.pad)
    with Pool(40) as p:
        rets = list(tqdm(p.imap(pad_data_, files), total=len(files)))


if __name__ == '__main__':
    main()
