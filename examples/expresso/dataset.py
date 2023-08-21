# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import torch
import random
import numpy as np
from librosa.util import normalize
from collections import Counter
from dataset import parse_speaker, load_audio, mel_spectrogram, MAX_WAV_VALUE

import os
from pathlib import Path


class MultiSpkrMultiStyleCodeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_files,
        segment_size,
        code_hop_size,
        n_fft,
        num_mels,
        hop_size,
        win_size,
        sampling_rate,
        fmin,
        fmax,
        fmax_loss=None,
        n_cache_reuse=1,
        device=None,
        input_file=None,
        pad=None,
        multispkr=None,
        multistyle=None,
        speakers=None,
        styles=None,
    ):

        random.seed(1234)

        self.audio_files, self.codes = training_files
        self.segment_size = segment_size
        self.code_hop_size = code_hop_size
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.multispkr = multispkr
        self.multistyle = multistyle
        self.pad = pad

        if self.multispkr:
            if self.multispkr != "from_input_file":
                self.spkr_names = [
                    parse_speaker(f, self.multispkr) for f in self.audio_files
                ]
            else:
                assert (
                    input_file is not None
                ), "input_file is required when multispkr=='from_input_file'"
                with open(input_file) as f:
                    self.spkr_names = [eval(line.strip())["spk"] for line in f]
                assert len(self.spkr_names) == len(self.audio_files)
            # Sort the speakers by occurences
            if speakers is None:
                speakers = [item[0] for item in Counter(self.spkr_names).most_common()]

            self.id_to_spkr = speakers
            self.spkr_to_id = {k: v for v, k in enumerate(self.id_to_spkr)}

        if self.multistyle:
            if self.multistyle != "from_input_file":
                self.style_names = [
                    parse_speaker(f, self.multistyle) for f in self.audio_files
                ]
            else:
                assert (
                    input_file is not None
                ), "input_file is required when multistyle=='from_input_file'"
                with open(input_file) as f:
                    self.style_names = [eval(line.strip())["style"] for line in f]
                assert len(self.style_names) == len(self.audio_files)
            # Sort the styles by occurences
            if styles is None:
                styles = [item[0] for item in Counter(self.style_names).most_common()]

            self.id_to_style = styles
            self.style_to_id = {k: v for v, k in enumerate(self.id_to_style)}

    def _sample_interval(self, seqs, seq_len=None):
        N = max([v.shape[-1] for v in seqs])
        if seq_len is None:
            seq_len = self.segment_size if self.segment_size > 0 else N

        hops = [N // v.shape[-1] for v in seqs]
        lcm = np.lcm.reduce(hops)

        # Randomly pickup with the batch_max_steps length of the part
        interval_start = 0
        interval_end = N // lcm - seq_len // lcm

        start_step = random.randint(interval_start, interval_end)

        new_seqs = []
        for i, v in enumerate(seqs):
            start = start_step * (lcm // hops[i])
            end = (start_step + seq_len // lcm) * (lcm // hops[i])
            new_seqs += [v[..., start:end]]

        return new_seqs

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_audio(filename)
            if sampling_rate != self.sampling_rate:
                import resampy

                audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

            if self.pad:
                padding = self.pad - (audio.shape[-1] % self.pad)
                audio = np.pad(audio, (0, padding), "constant", constant_values=0)
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        # Trim audio ending
        code_length = min(
            audio.shape[0] // self.code_hop_size, self.codes[index].shape[0]
        )
        code = self.codes[index][:code_length]
        audio = audio[: code_length * self.code_hop_size]
        assert (
            audio.shape[0] // self.code_hop_size == code.shape[0]
        ), "Code audio mismatch"

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            code = np.hstack([code, code])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        assert audio.size(1) >= self.segment_size, "Padding not supported!!"
        audio, code = self._sample_interval([audio, code])

        mel_loss = mel_spectrogram(
            audio,
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax_loss,
            center=False,
        )

        feats = {"code": code.squeeze()}
        if self.multispkr:
            feats["spkr"] = self._get_spkr(index)
        if self.multistyle:
            feats["style"] = self._get_style(index)

        return feats, audio.squeeze(0), str(filename), mel_loss.squeeze()

    def _get_spkr(self, idx):
        spkr_name = self.spkr_names[idx]
        spkr_id = torch.LongTensor([self.spkr_to_id[spkr_name]]).view(1).numpy()
        return spkr_id

    def _get_style(self, idx):
        style_name = self.style_names[idx]
        style_id = torch.LongTensor([self.style_to_id[style_name]]).view(1).numpy()
        return style_id

    def __len__(self):
        return len(self.audio_files)


class InferenceCodeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_code_file,
        name_parts=False,
        sampling_rate=None,
        multispkr=None,
        speakers=None,
        forced_speaker=None,
        random_speaker=False,
        random_speaker_subset=None,
        multistyle=None,
        styles=None,
        forced_style=None,
        random_style=False,
        random_style_subset=None,
    ):
        """
        The code file expects each line to be a dictionary
        {"audio": "filename.wav", "hubert": "1 2 2..", "spk": "01", "style": "03"}
        ..
        """

        random.seed(1234)

        if multispkr:
            assert speakers is not None, "speaker list expected for multispkr!"
            assert (
                forced_speaker is None or random_speaker is False
            ), "Cannot force speaker and choose random speaker at the same time"

            self.id_to_spkr = speakers
            self.spkr_to_id = {k: v for v, k in enumerate(self.id_to_spkr)}

            if random_speaker_subset is None or len(random_speaker_subset) == 0:
                random_speaker_subset = speakers

        if multistyle:
            assert styles is not None, "style list expected for multistyle!"
            assert (
                forced_style is None or random_style is False
            ), "Cannot force style and choose random style at the same time"

            self.id_to_style = styles
            self.style_to_id = {k: v for v, k in enumerate(self.id_to_style)}
            if random_style_subset is None or len(random_style_subset) == 0:
                random_style_subset = styles

        self.sampling_rate = sampling_rate
        self.multispkr = multispkr
        self.multistyle = multistyle

        self.audio_files = []
        self.codes = []
        self.spkr_names = []
        self.style_names = []
        self.output_file_names = []
        with open(input_code_file) as f:
            for line in f:
                content = eval(line)
                # Audio
                audio = content["audio"]
                self.audio_files.append(audio)

                # Code
                code = content["hubert"] if "hubert" in content else content["codes"]
                self.codes.append(code)

                # Speaker
                speaker = None
                if multispkr:
                    if forced_speaker:
                        speaker = forced_speaker
                    elif random_speaker:
                        speaker = random.choice(random_speaker_subset)
                    else:
                        assert "spk" in content, (
                            "Key 'spk' expected in input_code_file when "
                            "not using forced_speaker or random_speaker"
                        )
                        speaker = content["spk"]
                    assert speaker in speakers, (
                        f"Speaker '{speaker}' not in the list of speaker: {speakers}, "
                        "consider forced_speaker or random_speaker"
                    )
                self.spkr_names.append(speaker)

                # Style
                style = None
                if multistyle:
                    if forced_style:
                        style = forced_style
                    elif random_style:
                        style = random.choice(random_style_subset)
                    else:
                        assert "style" in content, (
                            "Key 'style' expected in input_code_file when "
                            "not using forced_style or random_style"
                        )
                        style = content["style"]
                    assert style in styles, (
                        f"Style '{style}' not in the list of style: {styles}, "
                        "consider forced_style or random_style"
                    )
                self.style_names.append(style)

                # Output filename
                if name_parts:
                    parts = Path(audio).parts
                    fname_out_name = os.path.splitext("_".join(parts[-3:]))[0]
                else:
                    fname_out_name = Path(audio).stem
                if multispkr:
                    fname_out_name = fname_out_name + f"_{speaker}"
                if multistyle:
                    fname_out_name = fname_out_name + f"_{style}"
                self.output_file_names.append(fname_out_name)

            print(f"Loaded {len(self.audio_files)} files from {input_code_file}!")

        if multispkr:
            if forced_speaker:
                print(f"Force speaker='{forced_speaker}'")
            elif random_speaker:
                print(f"Sample speaker randomly from: {random_speaker_subset}")
            else:
                print(f"Load default speaker from input_code_file")

        if multistyle:
            if forced_style:
                print(f"Force style='{forced_style}'")
            elif random_speaker:
                print(f"Sample style randomly from: {random_style_subset}")
            else:
                print(f"Load default style from input_code_file")

    def get_audio(self, path):
        if not os.path.exists(path) or self.sampling_rate is None:
            return None

        audio, sampling_rate = load_audio(path)
        if sampling_rate != self.sampling_rate:
            import resampy

            audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95

        return audio

    def __getitem__(self, index):
        filename = self.audio_files[index]

        audio = self.get_audio(filename)
        code = self.codes[index]
        speaker = self.spkr_names[index]
        style = self.style_names[index]
        out_filename = self.output_file_names[index]

        feats = {"code": np.array(list(map(int, code.split())))}
        if self.multispkr:
            feats["spkr"] = np.array([self.spkr_to_id[speaker]])
        if self.multistyle:
            feats["style"] = np.array([self.style_to_id[style]])

        return feats, audio, str(filename), out_filename

    def __len__(self):
        return len(self.audio_files)
