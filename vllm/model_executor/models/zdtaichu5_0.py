# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only ZDTaichu5.0 vision-language model."""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from vllm.config import VllmConfig
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.interfaces import SupportsEagle3, SupportsMRoPE
from vllm.model_executor.models.nano_nemotron_vl import (
    BaseNanoNemotronVLProcessor,
    NanoNemotronVLDummyInputsBuilder,
    NanoNemotronVLMultiModalProcessor,
    NanoNemotronVLProcessingInfo,
    NemotronH_Nano_VL_V2,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
)
from vllm.multimodal.processing.processor import PromptReplacement
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processors.zdtaichu5_0 import (
    IMAGE_CONTEXT,
    IMAGE_PROMPT_TARGET,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VIDEO_PROMPT_TARGET,
    VISION_END,
    VISION_START,
    ZDTaichu5_0_Processor,
    get_video_timing,
)

from .utils import _merge_multimodal_embeddings, maybe_prefix


class ZDTaichu5_0_ProcessingInfo(NanoNemotronVLProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> ZDTaichu5_0_Processor:
        return self.ctx.init_processor(
            ZDTaichu5_0_Processor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            video_token=self.get_video_token(),
            video_pruning_rate=self.get_video_pruning_rate(),
            max_model_len=self.ctx.model_config.max_model_len,
            **kwargs,
        )

    @property
    def supports_audio(self) -> bool:
        return False

    def get_video_token(self) -> str | None:
        # Video frames use the same image-context token as still images.
        return IMAGE_CONTEXT


class ZDTaichu5_0_DummyInputsBuilder(NanoNemotronVLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: dict[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        return IMAGE_PROMPT_TARGET * num_images + VIDEO_PROMPT_TARGET * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: dict[str, int],
        mm_options: dict[str, object],
        **kwargs: object,
    ) -> MultiModalDataDict:
        del kwargs
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        (target_width, target_height), _ = (
            self.info.get_dummy_image_size_and_max_tokens(mm_counts)
        )
        image_size = self.info.get_hf_config().force_image_size

        image_overrides = mm_options.get("image")
        video_overrides = mm_options.get("video")

        dummy_image = {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,  # type: ignore[arg-type]
            )
        }

        dummy_video: dict[str, object] = {}
        if self.info.supports_video:
            target_num_frames = self.info.get_num_frames_with_most_features(
                seq_len, mm_counts
            )
            dummy_video = {
                "video": self._get_dummy_videos(
                    width=image_size,
                    height=image_size,
                    num_frames=target_num_frames,
                    num_videos=num_videos,
                    overrides=video_overrides,  # type: ignore[arg-type]
                )
            }

        return {**dummy_image, **dummy_video}

    def get_dummy_image_data(
        self,
        seq_len: int,
        mm_counts: dict[str, int],
    ) -> dict[str, object]:
        proc = self.info.get_hf_processor()
        cfg = self.info.get_hf_config()
        image_size = getattr(cfg, "force_image_size", 512)
        num_images = mm_counts.get("image", 1)

        dummy_images = [
            Image.new("RGB", (image_size, image_size), color=255)
            for _ in range(num_images)
        ]
        out = proc(images=dummy_images, text=[IMAGE_PROMPT_TARGET * num_images])
        return {
            key: out[key]
            for key in ("pixel_values_flat", "image_num_patches", "image_grid_thw")
            if key in out
        }


class ZDTaichu5_0_MultiModalProcessor(NanoNemotronVLMultiModalProcessor):
    _IMAGE_CHAT_TEMPLATE_TARGET = IMAGE_PROMPT_TARGET
    _VIDEO_CHAT_TEMPLATE_TARGET = VIDEO_PROMPT_TARGET

    def _call_hf_processor(
        self,
        prompt,
        mm_data,
        mm_kwargs,
        tok_kwargs,
    ):
        # Force the dummy multimodal-budget path through the full text+mm
        # processor flow. The mm-only helper expects a Transformers
        # ProcessorMixin with `_merge_kwargs`, while this model uses vLLM's
        # lightweight inherited vLLM processor.
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _get_image_fields_config(self, hf_inputs):
        fields = super()._get_image_fields_config(hf_inputs)
        fields["image_grid_thw"] = MultiModalFieldConfig.batched(
            "image", keep_on_cpu=True
        )
        return fields

    def _get_video_fields_config(self, hf_inputs):
        fields = super()._get_video_fields_config(hf_inputs)
        fields["video_grid_thw"] = MultiModalFieldConfig.batched(
            "video", keep_on_cpu=True
        )
        return fields

    def _get_prompt_repl_image(self, mm_items, hf_processor, out_mm_data):
        update = super()._get_prompt_repl_image(mm_items, hf_processor, out_mm_data)
        return PromptReplacement(
            modality=update.modality,
            target=self._IMAGE_CHAT_TEMPLATE_TARGET,
            replacement=update.replacement,
        )

    def _get_prompt_repl_video(self, mm_items, hf_processor, out_mm_data):
        # The processor already resolved ``frames_indices`` and
        # ``frame_duration_ms`` from video metadata in ``_preprocess_video``;
        # reuse those values instead of renumbering sampled frames from zero,
        # which would anchor timestamps to the sampled-frame ordinal rather
        # than to the original video timeline.
        video_num_patches = out_mm_data.get("video_num_patches")
        frames_indices = out_mm_data.get("frames_indices")
        frame_duration_ms = out_mm_data.get("frame_duration_ms")
        if (
            video_num_patches is None
            or frames_indices is None
            or frame_duration_ms is None
        ):
            # Last-resort fallback for byte/decoded-video wrappers whose
            # processed data does not carry the timing fields.
            video_num_patches = []
            frames_indices = []
            frame_duration_ms = []
            for video, metadata in mm_items["video"]:
                num_frames = None
                if video is not None and hasattr(video, "shape"):
                    num_frames = int(video.shape[0])
                if num_frames is None:
                    frame_indices = metadata.get("frames_indices")
                    if frame_indices is not None:
                        num_frames = len(frame_indices)
                if num_frames is None:
                    raise ValueError(
                        "Unable to determine the number of frames for a video"
                    )
                video_num_patches.append(num_frames)
                indices, duration_ms = get_video_timing(metadata)
                frames_indices.append(indices)
                frame_duration_ms.append(duration_ms)

        if isinstance(video_num_patches, torch.Tensor):
            video_num_patches = video_num_patches.tolist()
        else:
            video_num_patches = list(video_num_patches)

        def get_video_replacement(item_idx: int):
            num_frames = int(video_num_patches[item_idx])
            indices = frames_indices[item_idx]
            if isinstance(indices, torch.Tensor):
                indices = indices.tolist()
            return hf_processor.get_video_repl(
                tokens_per_frame=[hf_processor.num_image_token] * num_frames,
                frames_indices=indices,
                frame_duration_ms=float(frame_duration_ms[item_idx]),
                tokenizer=hf_processor.tokenizer,
                img_start_token_ids=hf_processor._img_start_token_ids,
                img_end_token_ids=hf_processor._img_end_token_ids,
                img_context_token_ids=hf_processor._img_context_token_ids,
                video_temporal_patch_size=1,
            )

        return PromptReplacement(
            modality="video",
            target=self._VIDEO_CHAT_TEMPLATE_TARGET,
            replacement=get_video_replacement,
        )

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs,
        out_mm_kwargs,
    ):
        # The reference Transformers processor converts videos to image
        # frames. In that path ``mm_items`` has no ``video`` entry, while the
        # NanoNemotron parent would still unconditionally build a video
        # replacement and fail with KeyError. Build the prompt updates here so
        # video handling is enabled only when video items are actually present.
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        out_mm_data = out_mm_kwargs.get_data()
        updates = [self._get_prompt_repl_image(mm_items, hf_processor, out_mm_data)]
        if "video" in mm_items:
            updates.append(
                self._get_prompt_repl_video(mm_items, hf_processor, out_mm_data)
            )
        return updates


@MULTIMODAL_REGISTRY.register_processor(
    ZDTaichu5_0_MultiModalProcessor,
    info=ZDTaichu5_0_ProcessingInfo,
    dummy_inputs=ZDTaichu5_0_DummyInputsBuilder,
)
class ZDTaichu5_0_ForConditionalGeneration(NemotronH_Nano_VL_V2, SupportsMRoPE, SupportsEagle3):
    """C-RADIOv4-H vision tower with a Qwen3.5 hybrid language model."""

    packed_modules_mapping = dict(
        Qwen3_5ForConditionalGeneration.packed_modules_mapping
    )

    # The reference model can prune video tokens inside forward(). vLLM needs
    # model-specific M-RoPE recomputation for that optimization; keep the
    # non-pruned, numerically faithful path until it is implemented.
    supports_multimodal_pruning = False

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality.startswith("image"):
            return IMAGE_PROMPT_TARGET
        if modality.startswith("video"):
            return VIDEO_PROMPT_TARGET
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        config.norm_mean = getattr(config, "norm_mean", None) or list(IMAGENET_MEAN)
        config.norm_std = getattr(config, "norm_std", None) or list(IMAGENET_STD)
        config.use_thumbnail = getattr(config, "use_thumbnail", True)
        config.patch_size = getattr(
            config, "patch_size", config.vision_config.patch_size
        )
        # The reference model calls this nested config llm_config but also
        # aliases it to text_config. Preserve both forms for older checkpoints.
        if not hasattr(config, "text_config"):
            config.text_config = config.llm_config
        nn.Module.__init__(self)

        model_config = vllm_config.model_config
        image_size = config.force_image_size
        patch_size = config.patch_size
        self.patch_size = patch_size
        self.template = config.template
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.image_tag_type = config.image_tag_type
        # The reference processor treats every video frame as an image and
        # never enters the model's separate EVS video-pruning path.
        self.video_pruning_rate = None

        self.video_temporal_patch_size = 1

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=maybe_prefix(prefix, "language_model"),
            )
        llm_dtype = self.language_model.config.dtype
        assert isinstance(llm_dtype, torch.dtype)
        self.llm_dtype = llm_dtype

        with self._mark_tower_model(vllm_config, {"image", "video", "audio"}):
            self.vision_model = self.get_vit_model_from_radio_config(config).to(
                llm_dtype
            )

            vit_hidden_size = config.vit_hidden_size
            vision_projection_hidden_size = config.projector_hidden_size
            llm_hidden_size = config.text_config.hidden_size

            self.mlp1 = nn.Sequential(
                RMSNorm(
                    hidden_size=vit_hidden_size
                    * int(round(1 / self.downsample_ratio)) ** 2,
                    eps=1e-5,
                ),
                nn.Linear(
                    vit_hidden_size * int(round(1 / self.downsample_ratio)) ** 2,
                    vision_projection_hidden_size,
                    bias=False,
                ),
                ReLUSquaredActivation(),
                nn.Linear(vision_projection_hidden_size, llm_hidden_size, bias=False),
            ).to(llm_dtype)
            self.sound_encoder = None

        self.config = config
        self.model_config = vllm_config.model_config

        tokenizer = cached_tokenizer_from_config(self.model_config)
        self._img_start_token_ids = tokenizer.encode(
            VISION_START, add_special_tokens=False
        )
        self._img_end_token_ids = tokenizer.encode(VISION_END, add_special_tokens=False)
        self._img_context_token_ids = tokenizer.encode(
            IMAGE_CONTEXT, add_special_tokens=False
        )
        self.dynamic_resolution = BaseNanoNemotronVLProcessor.use_dynamic_resolution(
            config
        )

    def _parse_and_validate_image_input(self, **kwargs: object):
        kwargs.pop("image_grid_thw", None)
        return super()._parse_and_validate_image_input(**kwargs)

    def _parse_and_validate_video_input(self, **kwargs: object):
        kwargs.pop("video_grid_thw", None)
        return super()._parse_and_validate_video_input(**kwargs)

    def _create_final_video_embeddings(
        self,
        video_embeddings: torch.Tensor,
        num_tokens_per_frame: list[int],
        frames_indices: list[int],
        frame_duration_ms: int | float,
        video_temporal_patch_size: int = 1,
    ) -> torch.Tensor:
        tokenizer = cached_tokenizer_from_config(self.model_config)
        video_repl = ZDTaichu5_0_Processor.get_video_repl(
            tokens_per_frame=num_tokens_per_frame,
            frames_indices=frames_indices,
            frame_duration_ms=frame_duration_ms,
            tokenizer=tokenizer,
            img_start_token_ids=self._img_start_token_ids,
            img_end_token_ids=self._img_end_token_ids,
            img_context_token_ids=self._img_context_token_ids,
            video_temporal_patch_size=video_temporal_patch_size,
        )
        repl_token_ids = torch.tensor(video_repl.full, device=video_embeddings.device)
        embed_token_ids = torch.tensor(
            self._img_context_token_ids, device=video_embeddings.device
        )
        is_video_embed = torch.isin(repl_token_ids, embed_token_ids)
        text_embeddings = self.get_language_model().embed_input_ids(repl_token_ids)
        return _merge_multimodal_embeddings(
            inputs_embeds=text_embeddings,
            multimodal_embeddings=video_embeddings,
            is_multimodal=is_video_embed,
        )

    @staticmethod
    def _grid_from_feature(
        feature: MultiModalFeatureSpec,
        key: str,
    ) -> tuple[int, int, int] | None:
        if feature.data is None or key not in feature.data:
            return None
        value = feature.data[key].data
        tensor = torch.as_tensor(value).reshape(-1, 3)
        return tuple(int(x) for x in tensor[0].tolist())

    def _vision_positions(
        self,
        *,
        start_position: int,
        tile_rows: int,
        tile_cols: int,
        num_tokens: int,
    ) -> np.ndarray:
        tile_h = int(
            (self.config.force_image_size // self.patch_size) * self.downsample_ratio
        )
        tile_w = tile_h
        tokens_per_tile = tile_h * tile_w
        grid_tiles = tile_rows * tile_cols
        has_thumbnail = num_tokens > grid_tiles * tokens_per_tile

        tile_idx = np.arange(grid_tiles)
        tile_r = tile_idx // tile_cols
        tile_c = tile_idx % tile_cols
        local_idx = np.arange(tokens_per_tile)
        local_r = local_idx // tile_w
        local_c = local_idx % tile_w
        pos_h = (tile_r[:, None] * tile_h + local_r[None, :]).reshape(-1)
        pos_w = (tile_c[:, None] * tile_w + local_c[None, :]).reshape(-1)
        pos_t = np.full(pos_h.shape, start_position)

        if has_thumbnail:
            pos_t = np.concatenate((pos_t, np.full(tokens_per_tile, start_position)))
            pos_h = np.concatenate((pos_h, local_r * tile_rows))
            pos_w = np.concatenate((pos_w, local_c * tile_cols))

        positions = np.stack((pos_t, start_position + pos_h, start_position + pos_w))
        if positions.shape[1] != num_tokens:
            raise ValueError(
                "ZDTaichu5.0 image token/grid mismatch: "
                f"tokens={num_tokens}, grid=({tile_rows}, {tile_cols}), "
                f"computed={positions.shape[1]}"
            )
        return positions

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        image_token_id = self.config.img_context_token_id
        features = sorted(mm_features, key=lambda item: item.mm_position.offset)
        pieces: list[np.ndarray] = []
        token_cursor = 0
        current_position = 0

        def append_text(end: int) -> None:
            nonlocal token_cursor, current_position
            length = end - token_cursor
            if length > 0:
                positions = np.arange(length) + current_position
                pieces.append(np.broadcast_to(positions, (3, length)))
                current_position += length
                token_cursor = end

        for feature in features:
            start = feature.mm_position.offset
            end = start + feature.mm_position.length
            append_text(start)

            if feature.modality == "image":
                grid = self._grid_from_feature(feature, "image_grid_thw")
                if grid is None:
                    grid = (1, 1, 1)
                _, rows, cols = grid
                try:
                    run_start = input_tokens.index(image_token_id, start, end)
                except ValueError as exc:
                    raise ValueError(
                        "Image placeholder contains no image context tokens"
                    ) from exc
                append_text(run_start)
                run_end = run_start
                while run_end < end and input_tokens[run_end] == image_token_id:
                    run_end += 1
                pieces.append(
                    self._vision_positions(
                        start_position=current_position,
                        tile_rows=rows,
                        tile_cols=cols,
                        num_tokens=run_end - run_start,
                    )
                )
                tile_side = int(
                    (self.config.force_image_size // self.patch_size)
                    * self.downsample_ratio
                )
                current_position += max(rows * tile_side, cols * tile_side)
                token_cursor = run_end
                append_text(end)
                continue

            # ZDTaichu5.0 treats video as a sequence of independent image
            # frames. Each frame therefore gets the same 3D position layout
            # as an image tile, with a fresh position range after its header.
            cursor = start
            while cursor < end:
                try:
                    run_start = input_tokens.index(image_token_id, cursor, end)
                except ValueError:
                    append_text(end)
                    break
                append_text(run_start)
                run_end = run_start
                while run_end < end and input_tokens[run_end] == image_token_id:
                    run_end += 1
                pieces.append(
                    self._vision_positions(
                        start_position=current_position,
                        tile_rows=1,
                        tile_cols=1,
                        num_tokens=run_end - run_start,
                    )
                )
                tile_side = int(
                    (self.config.force_image_size // self.patch_size)
                    * self.downsample_ratio
                )
                current_position += tile_side
                token_cursor = run_end
                cursor = run_end

        append_text(len(input_tokens))
        positions = np.concatenate(pieces, axis=1).reshape(3, -1)
        delta = int(positions.max()) + 1 - len(input_tokens)
        return torch.from_numpy(positions), delta

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()
