"""Validation-video callback for the working-memory maze."""

import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from lightning.pytorch import Callback

from video_export import H264VideoWriter, add_canvas_padding

CUE_NAMES = ("red", "blue", "green")
ACTION_NAMES = ("forward", "left", "right")
CUE_BGR = {"red": (50, 50, 220), "blue": (240, 100, 50), "green": (50, 200, 50)}
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def _phase_name(frame, valid, transition, decision, memory):
    if not valid:
        return "padding"
    if not transition:
        return "terminal"
    if not memory:
        return "cue"
    if frame.sum() == 0:
        return "delay"
    if decision:
        return "decision"
    return "action"


def _to_rgb_uint8(normalized_pixels):
    pixels = normalized_pixels.detach().float().cpu()
    pixels = pixels * IMAGENET_STD + IMAGENET_MEAN
    return (
        pixels.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    )


class WMMazeValidationVideo(Callback):
    """Write one annotated MP4 and prediction CSV per validation epoch."""

    def __init__(self, every_n_epochs=1, fps=4.0, sample_index=0, padding=24):
        super().__init__()
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be at least 1")
        if padding < 0:
            raise ValueError("padding cannot be negative")
        self.every_n_epochs = every_n_epochs
        self.fps = fps
        self.sample_index = sample_index
        self.padding = padding
        self._written_epoch = None

    def on_validation_epoch_start(self, trainer, pl_module):
        self._written_epoch = None

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        epoch = trainer.current_epoch + 1
        if (
            trainer.sanity_checking
            or not trainer.is_global_zero
            or epoch % self.every_n_epochs != 0
            or self._written_epoch == epoch
            or self.sample_index >= batch["pixels"].size(0)
        ):
            return
        required = ("cue_logits", "act_logits")
        if not isinstance(outputs, dict) or any(key not in outputs for key in required):
            return
        self._write(epoch, batch, outputs)
        self._written_epoch = epoch

    def _write(self, epoch, batch, outputs, video_path=None):
        index = self.sample_index
        cue_logits = outputs["cue_logits"][index].detach().float().cpu()
        act_logits = outputs["act_logits"][index].detach().float().cpu()
        length = min(cue_logits.size(0), batch["pixels"].size(1))
        pixels = _to_rgb_uint8(batch["pixels"][index, :length])
        cue_true = batch["cue_color"][index, :length].detach().cpu().long().numpy()
        action_true = batch["action"][index, :length].detach().cpu().long()
        if action_true.ndim > 1 and action_true.size(-1) == 1:
            action_true = action_true.squeeze(-1)
        action_true = action_true.numpy()
        valid = batch["valid_mask"][index, :length].detach().cpu().bool().numpy()
        transition = (
            batch["transition_mask"][index, :length].detach().cpu().bool().numpy()
        )
        decision = batch["decision_mask"][index, :length].detach().cpu().bool().numpy()
        memory = batch["memory_mask"][index, :length].detach().cpu().bool().numpy()
        cue_prob = cue_logits.softmax(-1)
        act_prob = act_logits.softmax(-1)
        cue_pred = cue_prob.argmax(-1).numpy()
        action_pred = act_prob.argmax(-1).numpy()

        if video_path is None:
            root = Path(os.environ.get("STABLEWM_HOME", "data")) / "validation"
            video_path = root / f"epoch_{epoch:03d}_episode_{index:02d}.mp4"
        else:
            video_path = Path(video_path)
            if video_path.suffix.lower() != ".mp4":
                video_path = video_path.with_suffix(".mp4")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = video_path.with_suffix(".csv")
        height, width = pixels.shape[1:3]
        panel_width = 430
        canvas_width = width + panel_width + 2 * self.padding
        canvas_height = height + 2 * self.padding
        rows = []
        with H264VideoWriter(
            video_path, self.fps, (canvas_width, canvas_height)
        ) as writer:
            for step in np.flatnonzero(valid):
                true_cue_name = CUE_NAMES[cue_true[step]]
                pred_cue_name = CUE_NAMES[cue_pred[step]]
                true_action_name = ACTION_NAMES[action_true[step]]
                pred_action_name = ACTION_NAMES[action_pred[step]]
                phase = _phase_name(
                    pixels[step],
                    valid[step],
                    transition[step],
                    decision[step],
                    memory[step],
                )
                canvas = np.full((height, width + panel_width, 3), 28, dtype=np.uint8)
                canvas[:, :width] = cv2.cvtColor(pixels[step], cv2.COLOR_RGB2BGR)
                x = width + 18
                lines = (
                    f"Epoch: {epoch}  Step: {step:02d}",
                    f"Phase: {phase}",
                    f"Cue GT:   {true_cue_name}",
                    f"Cue pred: {pred_cue_name} ({cue_prob[step, cue_pred[step]]:.2f})",
                    f"Act GT:   {true_action_name}",
                    f"Act pred: {pred_action_name} ({act_prob[step, action_pred[step]]:.2f})",
                    f"Decision: {'yes' if decision[step] else 'no'}",
                )
                for line_index, line in enumerate(lines):
                    cv2.putText(
                        canvas,
                        line,
                        (x, 27 + line_index * 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (235, 235, 235),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.rectangle(
                    canvas,
                    (width + panel_width - 55, 64),
                    (width + panel_width - 20, 99),
                    CUE_BGR[true_cue_name],
                    -1,
                )
                cv2.rectangle(
                    canvas,
                    (width + panel_width - 55, 105),
                    (width + panel_width - 20, 140),
                    CUE_BGR[pred_cue_name],
                    -1,
                )
                if decision[step]:
                    border = (
                        (0, 200, 0)
                        if action_pred[step] == action_true[step]
                        else (0, 0, 230)
                    )
                    cv2.rectangle(canvas, (2, 2), (width - 3, height - 3), border, 4)
                writer.write(
                    add_canvas_padding(canvas, self.padding, color=(28, 28, 28))
                )
                rows.append(
                    {
                        "epoch": epoch,
                        "step": int(step),
                        "phase": phase,
                        "cue_true": true_cue_name,
                        "cue_pred": pred_cue_name,
                        "cue_confidence": float(cue_prob[step, cue_pred[step]]),
                        "cue_correct": int(cue_pred[step] == cue_true[step]),
                        "action_true": true_action_name,
                        "action_pred": pred_action_name,
                        "action_confidence": float(act_prob[step, action_pred[step]]),
                        "action_correct": int(action_pred[step] == action_true[step]),
                        "decision": int(decision[step]),
                    }
                )
        fieldnames = (
            "epoch",
            "step",
            "phase",
            "cue_true",
            "cue_pred",
            "cue_confidence",
            "cue_correct",
            "action_true",
            "action_pred",
            "action_confidence",
            "action_correct",
            "decision",
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(rows)
        print(f"Validation video: {video_path}")
        print(
            f"Validation video canvas: {canvas_width}x{canvas_height} "
            f"(padding={self.padding}px)"
        )
        print(f"Validation cue predictions: {csv_path}")
