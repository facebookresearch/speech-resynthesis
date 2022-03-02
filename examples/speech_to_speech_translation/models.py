# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import CodeGenerator, Generator


class VariancePredictor(nn.Module):
    def __init__(
        self,
        encoder_embed_dim,
        var_pred_hidden_dim,
        var_pred_kernel_size,
        var_pred_dropout
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                encoder_embed_dim, var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size,
                padding=(var_pred_kernel_size - 1) // 2
            ),
            nn.ReLU()
        )
        self.ln1 = nn.LayerNorm(var_pred_hidden_dim)
        self.dropout = var_pred_dropout
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                var_pred_hidden_dim, var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size, padding=1
            ),
            nn.ReLU()
        )
        self.ln2 = nn.LayerNorm(var_pred_hidden_dim)
        self.proj = nn.Linear(var_pred_hidden_dim, 1)

    def forward(self, x):
        # Input: B x T x C; Output: B x T
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(self.ln1(x), p=self.dropout, training=self.training)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(self.ln2(x), p=self.dropout, training=self.training)
        return self.proj(x).squeeze(dim=2)


def process_duration(code, code_feat):
    uniq_code_count = []
    uniq_code_feat = []
    for i in range(code.size(0)):
        _, count = torch.unique_consecutive(code[i, :], return_counts=True)
        if len(count) > 2:
            # remove first and last code as segment sampling may cause incomplete segment length
            uniq_code_count.append(count[1:-1])
            uniq_code_idx = count.cumsum(dim=0)[:-2]
        else:
            uniq_code_count.append(count)
            uniq_code_idx = count.cumsum(dim=0) - 1
        uniq_code_feat.append(code_feat[i, uniq_code_idx, :].view(-1, code_feat.size(2)))
    uniq_code_count = torch.cat(uniq_code_count)

    # collate feat
    max_len = max(feat.size(0) for feat in uniq_code_feat)
    out = uniq_code_feat[0].new_zeros((len(uniq_code_feat), max_len, uniq_code_feat[0].size(1)))
    mask = torch.arange(max_len).repeat(len(uniq_code_feat), 1)
    for i, v in enumerate(uniq_code_feat):
        out[i, : v.size(0)] = v
        mask[i, :] = mask[i, :] < v.size(0)

    return out, mask.bool(), uniq_code_count.float()


class DurationCodeGenerator(Generator):
    """
    Discrete unit-based HiFi-GAN vocoder with duration prediction
    (used in https://arxiv.org/abs/2107.05604)
    The current implementation only supports unit and speaker ID input and
    does not support F0 input.
    """
    def __init__(self, h):
        super().__init__(h)

        self.dict = nn.Embedding(h.num_embeddings, h.embedding_dim)
        self.f0 = h.get('f0', None)
        self.multispkr = h.get('multispkr', None)

        if self.multispkr:
            self.spkr = nn.Embedding(200, h.embedding_dim)

        self.dur_predictor = None
        if h.get('dur_prediction_weight', None):
            self.dur_predictor = VariancePredictor(**h.dur_predictor_params)


    def forward(self, **kwargs):
        x = self.dict(kwargs['code']).transpose(1, 2)

        dur_losses = 0.0
        if self.dur_predictor:
            if self.training:
                # assume input code is always full sequence
                uniq_code_feat, uniq_code_mask, dur = process_duration(
                    kwargs['code'], x.transpose(1, 2))
                log_dur_pred = self.dur_predictor(uniq_code_feat)
                log_dur_pred = log_dur_pred[uniq_code_mask]
                log_dur = torch.log(dur + 1)
                dur_losses = F.mse_loss(log_dur_pred, log_dur, reduction="mean")
            elif kwargs.get('dur_prediction', False):
                # assume input code can be unique sequence only in eval mode
                assert x.size(0) == 1, "only support single sample batch in inference"
                log_dur_pred = self.dur_predictor(x.transpose(1, 2))
                dur_out = torch.clamp(
                    torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1
                )
                # B x C x T
                x = torch.repeat_interleave(x, dur_out.view(-1), dim=2)

        if self.multispkr:
            spkr = self.spkr(kwargs['spkr']).transpose(1, 2)
            spkr = self._upsample(spkr, x.shape[-1])
            x = torch.cat([x, spkr], dim=1)

        return super().forward(x), dur_losses
