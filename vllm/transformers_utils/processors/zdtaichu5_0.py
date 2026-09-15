# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Processor helpers for ZDTaichu5.0 vision-language models."""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import regex as re
import torch
from PIL import Image

from vllm.multimodal.processing.processor import PromptUpdateDetails
from vllm.tokenizers.hf import HfTokenizer

from .nano_nemotron_vl import (
    NanoNemotronVLProcessor,
)

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_CONTEXT = "<|image_pad|>"
IMAGE_PROMPT_TARGET = VISION_START + IMAGE_CONTEXT + VISION_END
VIDEO_CONTEXT = "<|video_pad|>"
VIDEO_PROMPT_TARGET = VISION_START + VIDEO_CONTEXT + VISION_END

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _pil_to_nhwc_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB") if image.mode != "RGB" else image
    array = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(array).unsqueeze(0)


def _bicubic_resize_and_normalize(
    tensor: torch.Tensor,
    size: tuple[int, int],
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = tensor.permute(0, 3, 1, 2).to(dtype=torch.float32)
    tensor = torch.nn.functional.interpolate(
        tensor, size=size, mode="bicubic", align_corners=False, antialias=True
    )
    return ((tensor / 255.0 - norm_mean) / norm_std).to(dtype=dtype).contiguous()


def get_target_ratios(min_num_tiles: int, max_num_tiles: int) -> list[tuple[int, int]]:
    ratios = {
        (cols, rows)
        for n in range(min_num_tiles, max_num_tiles + 1)
        for cols in range(1, n + 1)
        for rows in range(1, n + 1)
        if min_num_tiles <= cols * rows <= max_num_tiles
    }
    return sorted(ratios, key=lambda ratio: ratio[0] * ratio[1])


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    *,
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio
    return best_ratio


def select_tile_grid(
    *,
    orig_width: int,
    orig_height: int,
    image_size: int,
    min_num_tiles: int,
    max_num_tiles: int,
) -> tuple[int, int]:
    """Return the (rows, columns) grid used by the reference processor."""
    src_pixels = orig_width * orig_height
    tile_pixels = image_size * image_size
    area_max_tiles = max(1, math.ceil(src_pixels / tile_pixels))
    effective_max = max(
        min_num_tiles,
        min(max_num_tiles, area_max_tiles),
    )

    target_ratios = get_target_ratios(min_num_tiles, effective_max)
    src_ar = orig_width / orig_height
    filtered = [
        ratio
        for ratio in target_ratios
        if (1.0 / 3.0) <= (ratio[0] / ratio[1]) / src_ar <= 3.0
    ]
    if filtered:
        target_ratios = filtered

    cols, rows = find_closest_aspect_ratio(
        src_ar,
        target_ratios,
        width=orig_width,
        height=orig_height,
        image_size=image_size,
    )
    return rows, cols


def count_tiles(
    *,
    orig_width: int,
    orig_height: int,
    image_size: int,
    min_num_tiles: int,
    max_num_tiles: int,
    use_thumbnail: bool,
) -> tuple[int, int, int]:
    rows, cols = select_tile_grid(
        orig_width=orig_width,
        orig_height=orig_height,
        image_size=image_size,
        min_num_tiles=min_num_tiles,
        max_num_tiles=max_num_tiles,
    )
    num_tiles = rows * cols
    if use_thumbnail and num_tiles != 1:
        num_tiles += 1
    return num_tiles, rows, cols


def image_to_pixel_values(
    image: Image.Image,
    *,
    image_size: int,
    min_num_tiles: int,
    max_num_tiles: int,
    use_thumbnail: bool,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    orig_width, orig_height = image.size
    _, rows, cols = count_tiles(
        orig_width=orig_width,
        orig_height=orig_height,
        image_size=image_size,
        min_num_tiles=min_num_tiles,
        max_num_tiles=max_num_tiles,
        use_thumbnail=use_thumbnail,
    )

    tensor = _pil_to_nhwc_tensor(image)
    resized = _bicubic_resize_and_normalize(
        tensor,
        size=(rows * image_size, cols * image_size),
        norm_mean=norm_mean,
        norm_std=norm_std,
        dtype=dtype,
    )
    batch, channels, height, width = resized.shape
    patches = (
        resized.reshape(
            batch,
            channels,
            rows,
            image_size,
            cols,
            image_size,
        )
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(rows * cols, channels, image_size, image_size)
    )
    if use_thumbnail and rows * cols != 1:
        thumbnail = _bicubic_resize_and_normalize(
            tensor,
            size=(image_size, image_size),
            norm_mean=norm_mean,
            norm_std=norm_std,
            dtype=dtype,
        )
        patches = torch.cat((patches, thumbnail), dim=0)

    return patches, (1, rows, cols)


def get_video_timing(metadata: dict[str, Any]) -> tuple[list[int], float]:
    """Return the (frames_indices, frame_duration_ms) for video ``metadata``.

    ``frames_indices`` preserves the loader's original-frame indices so
    timestamps stay anchored to the source video timeline even when only a
    subset of frames was sampled.
    """
    indices = metadata["frames_indices"]
    if indices is None:
        raise ValueError("Video metadata must provide a frames_indices")
    fps = metadata["fps"]
    if fps is None or fps <= 0:
        raise ValueError("Video metadata must provide a positive fps")
    return [int(i) for i in indices], 1000.0 / float(fps)


class ZDTaichu5_0_Processor(NanoNemotronVLProcessor):
    """vLLM processor matching ``ZDTaichu5_0_Processor``."""

    def __init__(
        self,
        config,
        tokenizer: HfTokenizer,
        *,
        max_model_len: int,
        max_num_tiles: int | None = None,
        video_token: str | None = None,
        video_pruning_rate: float | None = None,
        use_audio_in_video: bool = False,
    ) -> None:
        # The values live in preprocessor_config.json in the reference repo,
        # rather than in the top-level model config.
        config.norm_mean = getattr(config, "norm_mean", None) or list(IMAGENET_MEAN)
        config.norm_std = getattr(config, "norm_std", None) or list(IMAGENET_STD)
        config.use_thumbnail = getattr(config, "use_thumbnail", True)
        config.patch_size = getattr(
            config, "patch_size", config.vision_config.patch_size
        )

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            max_num_tiles=max_num_tiles or getattr(config, "max_dynamic_patch", 12),
            video_token=video_token,
            video_pruning_rate=None,
        )
        del use_audio_in_video, video_pruning_rate
        self.video_temporal_patch_size = 1
        self.dtype: torch.dtype = getattr(config, "dtype", torch.float32)
        self.min_num_tiles = getattr(config, "min_dynamic_patch", 1)
        self._img_start_token_ids = tokenizer.encode(
            VISION_START, add_special_tokens=False
        )
        self._img_end_token_ids = tokenizer.encode(VISION_END, add_special_tokens=False)
        self._img_context_token_ids = tokenizer.encode(
            IMAGE_CONTEXT, add_special_tokens=False
        )

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(IMAGE_CONTEXT)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        max_num_tiles: int,
    ) -> int:
        num_tiles, _, _ = count_tiles(
            orig_width=image_width,
            orig_height=image_height,
            image_size=self.image_size,
            min_num_tiles=self.min_num_tiles,
            max_num_tiles=max_num_tiles,
            use_thumbnail=self.use_thumbnail,
        )
        return num_tiles * self.num_image_token

    def _images_to_pixel_values_and_grids(
        self,
        images: list[Image.Image],
        max_num_tiles: int,
    ) -> tuple[list[torch.Tensor], list[tuple[int, int, int]]]:
        pixel_values: list[torch.Tensor] = []
        grids: list[tuple[int, int, int]] = []
        for image in images:
            image_pixels, image_grid = image_to_pixel_values(
                image,
                image_size=self.image_size,
                min_num_tiles=self.min_num_tiles,
                max_num_tiles=max_num_tiles,
                use_thumbnail=self.use_thumbnail,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
                dtype=self.dtype,
            )
            pixel_values.append(image_pixels)
            grids.append(image_grid)
        return pixel_values, grids

    def _images_to_pixel_values_lst(
        self,
        images: list[Image.Image],
        max_num_tiles: int,
    ) -> list[torch.Tensor]:
        pixel_values, _ = self._images_to_pixel_values_and_grids(images, max_num_tiles)
        return pixel_values

    def _videos_to_pixel_values_lst(
        self,
        videos: list[npt.NDArray],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> list[torch.Tensor]:
        """Process video frames through the fixed-size image pipeline."""
        pixel_values: list[torch.Tensor] = []
        for video in videos:
            frame_values = [
                image_to_pixel_values(
                    Image.fromarray(frame),
                    image_size=self.image_size,
                    min_num_tiles=1,
                    max_num_tiles=1,
                    use_thumbnail=self.use_thumbnail,
                    norm_mean=self.norm_mean,
                    norm_std=self.norm_std,
                    dtype=dtype,
                )[0]
                for frame in video
            ]
            pixel_values.append(torch.cat(frame_values))
        return pixel_values

    def _preprocess_image(
        self,
        text: list[str],
        images: list[Image.Image],
        max_num_tiles: int,
    ) -> tuple[list[str], dict[str, Any]]:
        if not images:
            return text, {}
        if self.dynamic_tiler is not None:
            raise ValueError(
                "ZDTaichu5.0 uses fixed 512px InternVL-style tiles"
            )

        pixel_values, grids = self._images_to_pixel_values_and_grids(
            images, max_num_tiles
        )
        image_num_patches = torch.tensor(
            [len(item) for item in pixel_values],
            dtype=torch.long,
        )
        image_inputs = {
            "pixel_values_flat": (
                torch.cat(pixel_values) if len(pixel_values) > 1 else pixel_values[0]
            ),
            "image_num_patches": image_num_patches,
            "image_grid_thw": torch.tensor(grids, dtype=torch.long),
        }

        if len(text) != 1:
            raise ValueError("Only a single prompt batch is supported")
        replaced_text = self._replace_image_placeholders(
            text[0], image_num_patches.tolist()
        )
        return [replaced_text], image_inputs

    def _replace_image_placeholders(
        self,
        text: str,
        image_num_patches: list[int],
    ) -> str:
        parts = [
            part
            for part in re.split(rf"({re.escape(IMAGE_PROMPT_TARGET)})", text)
            if part
        ]
        placeholder_indices = [
            index for index, part in enumerate(parts) if part == IMAGE_PROMPT_TARGET
        ]
        if len(placeholder_indices) != len(image_num_patches):
            raise ValueError(
                "Number of image placeholders does not match images: "
                f"placeholders={len(placeholder_indices)}, images={len(image_num_patches)}"
            )
        for part_index, num_patches in zip(
            placeholder_indices,
            image_num_patches,
            strict=True,
        ):
            feature_size = self.num_image_token * num_patches
            parts[part_index] = self.get_image_repl(feature_size, num_patches).full
        return "".join(parts)

    def _preprocess_video(
        self,
        text: list[str],
        videos: list[tuple[npt.NDArray, dict[str, Any]]],
    ) -> tuple[list[str], dict[str, Any]]:
        if not videos:
            return text, {}

        pixel_values = self._videos_to_pixel_values_lst(
            [video for video, _ in videos],
            dtype=self.dtype,
        )
        num_frames = [len(item) for item in pixel_values]
        # The video loader may have uniformly sampled a subset of the source
        # frames.  Preserve those source indices instead of renumbering them;
        # otherwise timestamps are computed relative to the sampled-frame
        # ordinal rather than the original video timeline.
        frames_indices = []
        frame_duration_ms = []
        for _, metadata in videos:
            indices, duration_ms = get_video_timing(metadata)
            frames_indices.append(indices)
            frame_duration_ms.append(duration_ms)
        video_inputs = {
            "pixel_values_flat_video": (
                torch.cat(pixel_values) if len(pixel_values) > 1 else pixel_values[0]
            ),
            "video_num_patches": torch.tensor(num_frames, dtype=torch.long),
            "frames_indices": frames_indices,
            "frame_duration_ms": torch.tensor(frame_duration_ms, dtype=torch.float64),
            "video_grid_thw": torch.tensor(
                [(count, 1, 1) for count in num_frames],
                dtype=torch.long,
            ),
        }

        for count, indices, duration_ms in zip(
            num_frames,
            frames_indices,
            frame_duration_ms,
            strict=True,
        ):
            replacement = self.get_video_repl(
                tokens_per_frame=[self.num_image_token] * count,
                frames_indices=indices,
                frame_duration_ms=duration_ms,
                tokenizer=self.tokenizer,
                img_start_token_ids=self._img_start_token_ids,
                img_end_token_ids=self._img_end_token_ids,
                img_context_token_ids=self._img_context_token_ids,
                video_temporal_patch_size=1,
            )
            replacement_text = self.tokenizer.decode(
                replacement.full,
                skip_special_tokens=False,
            )
            text = [
                item.replace(VIDEO_PROMPT_TARGET, replacement_text, 1)
                for item in text
            ]

        return text, video_inputs

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: int | None,
    ) -> PromptUpdateDetails[str]:
        del num_patches
        repl_full = f"{VISION_START}{IMAGE_CONTEXT * feature_size}{VISION_END}"
        return PromptUpdateDetails.select_text(repl_full, IMAGE_CONTEXT)

    @classmethod
    def get_video_repl(cls, **kwargs) -> PromptUpdateDetails[list[int]]:
        replacement = super().get_video_repl(**kwargs)
        tokenizer: HfTokenizer = kwargs["tokenizer"]
        prefix = tokenizer.encode(
            "This is a video:\n",
            add_special_tokens=False,
        )
        suffix = tokenizer.encode("\n", add_special_tokens=False)
        return PromptUpdateDetails.from_seq(prefix + replacement.full + suffix)
