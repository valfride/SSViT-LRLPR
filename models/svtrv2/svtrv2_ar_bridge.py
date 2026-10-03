import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.cppd.svtrnet import SVTRNet
from models.ote.nrtr_decoder import NRTRDecoder


class SVTRv2ARBaseline(nn.Module):
    """SVTRv2-style visual encoder with the OpenOCR NRTR autoregressive decoder.

    The architecture follows OpenOCR's `configs/rec/nrtr/svtrv2_nrtr.yml`,
    adapted only to the 32x96 LRLPR input and seven-character Brazilian plate
    vocabulary.  Temporal inference uses AR-aware BJP/product-rule fusion:
    every frame is conditioned on the same fused prefix at each decoding step.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=39,
        max_len=7,
        **kwargs,
    ):
        super().__init__()

        if num_classes != 39:
            raise ValueError(
                "SVTRv2-AR expects 39 decoder symbols: "
                "EOS + 36 alphanumerics + BOS + PAD."
            )

        # OpenOCR svtrv2_nrtr visual encoder, resized from 32x128 to 32x96.
        # The original configuration uses 18 blocks with a Conv/Global mixer
        # schedule and returns 384-dimensional visual tokens.
        self.encoder = SVTRNet(
            img_size=[32, 96],
            in_channels=in_channels,
            out_char_num=max_len,
            out_channels=256,
            patch_merging="Conv",
            embed_dim=[128, 256, 384],
            depth=[6, 6, 6],
            num_heads=[4, 8, 12],
            mixer=(
                ["Conv"] * 6
                + ["Conv"] * 2
                + ["Global"] * 4
                + ["Global"] * 6
            ),
            local_mixer=[[5, 5], [5, 5], [5, 5]],
            last_stage=False,
            prenorm=True,
        )

        # OpenOCR NRTR AR decoder.  With seven plate characters we decode
        # seven characters plus one EOS step.
        self.decoder = NRTRDecoder(
            in_channels=384,
            out_channels=num_classes,
            num_encoder_layers=-1,
            beam_size=0,
            num_decoder_layers=2,
            nhead=12,
            max_len=max_len + 1,
        )

        self.max_len = int(max_len)
        self.decode_steps = self.max_len + 1

    def _next_token_log_probs(self, memory, prefix):
        """Return log p(y_t | shared_prefix, frame) for one AR step."""
        tgt = self.decoder.embedding(prefix)
        tgt = self.decoder.positional_encoding(tgt)
        tgt_mask = self.decoder.generate_square_subsequent_mask(
            tgt.shape[1],
            memory.device,
        )

        for decoder_layer in self.decoder.decoder:
            tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)

        next_logits = self.decoder.tgt_word_prj(tgt[:, -1, :])
        return F.log_softmax(next_logits, dim=-1)

    def bjp_decode(self, flat_images, batch_size, frames):
        """Autoregressive BJP decoding with one prefix shared across frames.

        For each output step, all F frames evaluate the same current prefix.
        Their next-token log-probabilities are summed (product rule), the fused
        token is selected greedily, and that token becomes the shared prefix
        for the next step.
        """
        if frames < 1:
            raise ValueError(f"frames must be >= 1, got {frames}")
        if flat_images.shape[0] != batch_size * frames:
            raise ValueError(
                "flat_images batch does not match batch_size * frames: "
                f"{flat_images.shape[0]} != {batch_size} * {frames}"
            )

        memory_flat = self.encoder(flat_images)
        if memory_flat.ndim != 3:
            raise ValueError(
                "SVTRv2-AR encoder must return [B*F, tokens, channels], "
                f"got {tuple(memory_flat.shape)}"
            )

        token_count = memory_flat.shape[1]
        channels = memory_flat.shape[2]
        memory = memory_flat.reshape(
            batch_size,
            frames,
            token_count,
            channels,
        )

        prefix = torch.full(
            (batch_size, self.decode_steps + 1),
            self.decoder.ignore_index,
            dtype=torch.long,
            device=flat_images.device,
        )
        prefix[:, 0] = self.decoder.bos

        fused_steps = []
        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=flat_images.device,
        )

        for step in range(self.decode_steps):
            current_prefix = prefix[:, : step + 1]
            frame_log_probs = []

            for frame_idx in range(frames):
                frame_log_probs.append(
                    self._next_token_log_probs(
                        memory[:, frame_idx],
                        current_prefix,
                    )
                )

            # BJP/product rule in log space. Division by F is unnecessary for
            # argmax decoding and would not change the normalized distribution.
            fused_log_probs = torch.stack(frame_log_probs, dim=1).sum(dim=1)
            fused_steps.append(fused_log_probs.unsqueeze(1))

            next_token = fused_log_probs.argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.decoder.eos),
                next_token,
            )
            prefix[:, step + 1] = next_token
            finished = finished | next_token.eq(self.decoder.eos)

            if finished.all():
                break

        return torch.cat(fused_steps, dim=1)

    def forward(self, x, tgt=None, **kwargs):
        memory = self.encoder(x)

        # Teacher-forced AR training uses BOS + seven characters + EOS.
        if tgt is not None:
            logits = self.decoder.forward_train(memory, tgt)
        else:
            # Single-frame/native AR inference. Multi-frame paper evaluation
            # calls bjp_decode() instead so that all frames share one prefix.
            logits = self.decoder.forward_test(memory)

        return {
            "logits": logits,
            "attn_maps": getattr(self.decoder, "attn_maps", None),
            "latent_lr": None,
            "z_vector": None,
        }


class SVTRv2ARLoss(nn.Module):
    """OpenOCR-style AR cross entropy with label smoothing."""

    def __init__(self, ignore_index=38, label_smoothing=0.1):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)

    def forward(self, preds, targets):
        logits = preds["logits"]
        # targets = [BOS, c1, ..., c7, EOS]; logits predict [c1, ..., c7, EOS].
        target_next = targets[:, 1 : 1 + logits.shape[1]]
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_next.reshape(-1),
            reduction="mean",
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )


@register("SVTRV2_AR_BASELINE")
def make(**kwargs):
    return SVTRv2ARBaseline(**kwargs)
