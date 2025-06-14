import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from utils.attn_utils import fn_get_topk, fn_get_otsu_mask, fn_show_attention_plus
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
import matplotlib.pyplot as plt
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from pytorch_metric_learning import distances, losses
from torch.nn import functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
import os
logger = logging.get_logger()


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionAttendAndContrastPipeline

        >>> pipe = StableDiffusionAttendAndContrastPipeline.from_pretrained(
        ...     "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
        ... ).to("cuda")


        >>> prompt = "a cat and a frog"

        >>> # use get_indices function to find out indices of the tokens you want to alter
        >>> pipe.get_indices(prompt)
        {0: '<|startoftext|>', 1: 'a</w>', 2: 'cat</w>', 3: 'and</w>', 4: 'a</w>', 5: 'frog</w>', 6: '<|endoftext|>'}

        >>> token_indices = [2, 5]
        >>> seed = 6141
        >>> generator = torch.Generator("cuda").manual_seed(seed)

        >>> images = pipe(
        ...     prompt=prompt,
        ...     token_indices=token_indices,
        ...     guidance_scale=7.5,
        ...     generator=generator,
        ...     num_inference_steps=50,
        ...     max_iter_to_alter=25,
        ... ).images

        >>> image = images[0]
        >>> image.save(f"../images/{prompt}_{seed}.png")
        ```
"""


def fn_get_topk(attention_map, K=1):
    H, W = attention_map.size()
    attention_map_detach = attention_map.view(H * W) # .detach()
    topk_value_list, topk_index = attention_map_detach.topk(K, dim=0, largest=True, sorted=True)
    topk_coord_list = []
    for i in range(len(topk_index)):
        index = topk_index[i].cpu().numpy()
        coord = index // W, index % W
        topk_coord_list.append(coord)
    return topk_value_list, topk_coord_list


class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"

        if self.cur_att_layer >= 0:
            if attn.shape[1] == np.prod(self.attn_res):
                self.step_store[key].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def aggregate_attention(self, from_where: List[str], is_cross: bool = True) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.get_average_attention()

        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                cross_maps = item.reshape(-1, self.attn_res[0], self.attn_res[1], item.shape[-1])
                out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res


class AttentionProcessor:
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # only need to store attention maps during the Attend and Excite process
        if attention_probs.requires_grad:
            self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class ConformPipeline(DiffusionPipeline, TextualInversionLoaderMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion and Attend-and-Contrast.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(
                    image, output_type="pil"
                )
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(
                feature_extractor_input, return_tensors="pt"
            ).to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        indices,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        indices_is_list_list_ints = (
            isinstance(indices, list)
            and isinstance(indices[0], list)
            and isinstance(indices[0][0], int)
        )

        indices_is_list_list_list_ints = (
            isinstance(indices, list)
            and isinstance(indices[0], list)
            and isinstance(indices[0][0], list)
            and isinstance(indices[0][0][0], int)
        )

        if not indices_is_list_list_ints and not indices_is_list_list_list_ints:
            raise TypeError(
                "`indices` must be a list of a list of ints or a list of a list of a list of ints."
            )

        if indices_is_list_list_ints:
            indices_batch_size = 1
        else:
            indices_batch_size = len(indices)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if indices_batch_size != prompt_batch_size:
            raise ValueError(
                f"indices batch size must be same as prompt batch size. indices batch size: {indices_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents


    @staticmethod
    def _get_max_attention_per_token(
        attention_maps: torch.Tensor,
        indices: List[int],
        do_smoothing: bool = True,
        smoothing_kernel_size: int = 3,
        smoothing_sigma: float = 0.5,
    ) -> List[float]:
        """Computes the maximum attention value for each token."""
        # Shift indices since we removed the first token
        indices = [index - 1 for index in indices]

        # Extract the maximum values
        max_indices_list = []
        for i in indices:
            image = attention_maps[:, :, i]
            if do_smoothing:
                smoothing = GaussianSmoothing(
                    kernel_size=smoothing_kernel_size, sigma=smoothing_sigma
                ).to(attention_maps.device)
                input = F.pad(
                    image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                )
                image = smoothing(input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())
        return max_indices_list


    @staticmethod
    def _compute_contrastive_loss(
        t: int,
        cross_attention_maps: torch.Tensor,
        self_attention_maps: torch.Tensor,
        attention_maps_t_plus_one: Optional[torch.Tensor],
        token_groups: List[List[int]],
        loss_type: str,
        temperature: float = 0.07,
        do_smoothing: bool = True,
        smoothing_kernel_size: int = 3,
        smoothing_sigma: float = 0.5,
        softmax_normalize: bool = True,
        softmax_normalize_attention_maps: bool = False,
    ) -> torch.Tensor:
        """Computes the attend-and-contrast loss using the maximum attention value for each token."""

        attention_for_text = cross_attention_maps[:, :, 1:-1]
        attention_res=attention_for_text.shape[0]
        if softmax_normalize:
            #attention_for_text *= 10
            attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        attention_for_text_t_plus_one = None
        if attention_maps_t_plus_one is not None:
            attention_for_text_t_plus_one = attention_maps_t_plus_one[:, :, 1:-1]
            if softmax_normalize:
                #attention_for_text_t_plus_one *= 10
                attention_for_text_t_plus_one = torch.nn.functional.softmax(
                    attention_for_text_t_plus_one, dim=-1
                )
        #print(token_groups)
        indices_to_clases = {}
        for c, group in enumerate(token_groups):
            for obj in group:
                indices_to_clases[obj] = c
        # print("indices_to_classes", indices_to_classes)
        classes = []
        embeddings = []
        cross_attn_top_k_values = []
        self_attn_maps_list = []
        cross_attn_maps_list = []
        self_classes = []
        for ind, c in indices_to_clases.items():
            # print(ind ,c)
            classes.append(c)
            self_classes.append(c)
            # Shift indices since we removed the first token
            embedding = attention_for_text[:, :, ind - 1]
            if do_smoothing:
                smoothing = GaussianSmoothing(
                    kernel_size=smoothing_kernel_size, sigma=smoothing_sigma
                ).to(attention_for_text.device)
                input = F.pad(
                    embedding.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
                )
                embedding = smoothing(input).squeeze(0).squeeze(0)
            cross_attn_maps_list.append(embedding)
            topk_value, topk_coord = fn_get_topk(embedding, K=16)
            mean_tensor = torch.mean(topk_value, dim=0)
            cross_attn_top_k_values.append(mean_tensor)

            self_attn_map_cur_list=[]
            for i in range(len(topk_value)):
                self_attn_map_cur_token = topk_value[i]*self_attention_maps[topk_coord[i][0], topk_coord[i][1]]
                self_attn_map_cur_token = self_attn_map_cur_token.view(attention_res,attention_res).contiguous()
                self_attn_map_cur_list.append(self_attn_map_cur_token)
            self_attn_map_cur_token = torch.stack(self_attn_map_cur_list, dim=0).to(attention_for_text.device)
            self_attn_map_cur_token = torch.mean(self_attn_map_cur_token, dim=0)
            self_attn_maps_list.append(self_attn_map_cur_token)

            #self_embedding = self_attn_map_cur_token.view(-1)
            #classes.append(c)
            #embeddings.append(self_embedding)
            embedding = embedding.view(-1)
            if softmax_normalize_attention_maps:
                embedding *= 100
                embedding = torch.nn.functional.softmax(embedding)

            embeddings.append(embedding)


            if attention_for_text_t_plus_one is not None:
                classes.append(c)
                # Shift indices since we removed the first token
                embedding = attention_for_text_t_plus_one[:, :, ind - 1]
                if do_smoothing:
                    smoothing = GaussianSmoothing(
                        kernel_size=smoothing_kernel_size, sigma=smoothing_sigma
                    ).to(attention_for_text.device)
                    input = F.pad(
                        embedding.unsqueeze(0).unsqueeze(0),
                        (1, 1, 1, 1),
                        mode="reflect",
                    )
                    embedding = smoothing(input).squeeze(0).squeeze(0)
                embedding = embedding.view(-1)

                if softmax_normalize_attention_maps:
                    embedding *= 100
                    embedding = torch.nn.functional.softmax(embedding)
                embeddings.append(embedding)

        classes = torch.tensor(classes).to(attention_for_text.device)
        embeddings = torch.stack(embeddings, dim=0).to(attention_for_text.device)
        #print(classes.shape,embeddings.shape,token_groups)
        self_cross_attn_loss=0
        self_conflict_loss=0
        for i in range(len(self_classes)):
            for j in range(len(self_classes)):
                if self_classes[i]==self_classes[j]: continue
                else:
                    self_cross_attn_loss = self_cross_attn_loss + torch.min(self_attn_maps_list[i],cross_attn_maps_list[j]).sum() /torch.sum(self_attn_maps_list[i]+cross_attn_maps_list[j])
                    self_conflict_loss = self_conflict_loss+ torch.min(self_attn_maps_list[i],self_attn_maps_list[j]).sum() /torch.sum(self_attn_maps_list[i]+self_attn_maps_list[j])

        if loss_type == "ntxent_contrastive":
            if len(token_groups) > 0 and len(token_groups[0]) > 1:
                loss_fn = losses.NTXentLoss(temperature=temperature)
            else:
                loss_fn = losses.ContrastiveLoss(
                    distance=distances.CosineSimilarity(), pos_margin=1, neg_margin=0
                )
        elif loss_type == "ntxent":
            loss_fn = losses.NTXentLoss(temperature=temperature)
        elif loss_type == "contrastive":
            loss_fn = losses.ContrastiveLoss(
                distance=distances.CosineSimilarity(), pos_margin=1, neg_margin=0
            )
        else:
            raise ValueError(f"loss_fn {loss_type} not supported")

        cross_attn_loss_list = [max(0 * curr_max, 1.0 - curr_max) for curr_max in cross_attn_top_k_values]
        cross_attn_loss = max(cross_attn_loss_list)

        # if t==981 and cross_attn_loss>0.5: loss=cross_attn_loss
        # else: loss = cross_attn_loss + 0.1*self_cross_attn_loss +  0.1*self_conflict_loss
        loss=loss_fn(embeddings, classes)
        # print(loss.item())
        return loss

    @staticmethod
    def _update_latent(
        latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step(
        self,
        latents: torch.Tensor,
        token_groups: List[List[int]],
        loss: torch.Tensor,
        text_embeddings: torch.Tensor,
        step_size: float,
        t: int,
        refinement_steps: int = 20,
        do_smoothing: bool = True,
        smoothing_kernel_size: int = 3,
        smoothing_sigma: float = 0.5,
        temperature: float = 0.07,
        softmax_normalize: bool = True,
        softmax_normalize_attention_maps: bool = False,
        attention_maps_t_plus_one: Optional[torch.Tensor] = None,
        loss_fn: str = "ntxent",
    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent code
        according to our loss objective until the given threshold is reached for all tokens.
        """
        for iteration in range(refinement_steps):
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
            self.unet.zero_grad()

            # Get max activation value for each subject token
            cross_attention_maps = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"),is_cross = True)
            self_attention_maps = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"),is_cross = False)
            loss = self._compute_contrastive_loss(
                t=t,
                cross_attention_maps=cross_attention_maps,
                self_attention_maps=self_attention_maps,
                attention_maps_t_plus_one=attention_maps_t_plus_one,
                token_groups=token_groups,
                loss_type=loss_fn,
                do_smoothing=do_smoothing,
                temperature=temperature,
                smoothing_kernel_size=smoothing_kernel_size,
                smoothing_sigma=smoothing_sigma,
                softmax_normalize=softmax_normalize,
                softmax_normalize_attention_maps=softmax_normalize_attention_maps,
            )

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        _ = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        cross_attention_maps = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"),is_cross = True)
        self_attention_maps = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"), is_cross=False)

        loss = self._compute_contrastive_loss(
            t=t,
            cross_attention_maps=cross_attention_maps,
            self_attention_maps=self_attention_maps,
            attention_maps_t_plus_one=attention_maps_t_plus_one,
            token_groups=token_groups,
            loss_type=loss_fn,
            do_smoothing=do_smoothing,
            temperature=temperature,
            smoothing_kernel_size=smoothing_kernel_size,
            smoothing_sigma=smoothing_sigma,
            softmax_normalize=softmax_normalize,
            softmax_normalize_attention_maps=softmax_normalize_attention_maps,
        )
        return loss, latents

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttentionProcessor(
                attnstore=self.attention_store, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str,
        model_choice: str,
        token_groups: List[List[int]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_iter_to_alter: int = 25,
        refinement_steps: int = 20,
        iterative_refinement_steps: List[int] = [0, 10, 20],
        scale_factor: int = 20,
        attn_res: Optional[Tuple[int, int]] = (16, 16),
        steps_to_save_attention_maps: Optional[List[int]] = None,
        do_smoothing: bool = True,
        smoothing_kernel_size: int = 3,
        smoothing_sigma: float = 0.5,
        temperature: float = 0.5,
        softmax_normalize: bool = True,
        softmax_normalize_attention_maps: bool = False,
        add_previous_attention_maps: bool = True,
        previous_attention_map_anchor_step: Optional[int] = None,
        loss_fn: str = "ntxent",
        result_root: str = '',
        seed: int = 0,
        
        run_bon: bool = True,
        num_initial_noises: int = 1,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            token_indices (`List[int]`):
                The token indices to alter with attend-and-contrast.


        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            token_groups,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        if run_bon:
            latents_list = []
            for _ in range(num_initial_noises):
                lat_ = self.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    None,
                )
                latents_list.append(lat_)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))
        self.attention_store = AttentionStore(attn_res)
        self.register_attention_control()

        # default config for step size from original repo
        scale_range = np.linspace(1.0, 0.5, len(self.scheduler.timesteps))
        step_size = scale_factor * np.sqrt(scale_range)

        text_embeddings = (
            prompt_embeds[batch_size * num_images_per_prompt :]
            if do_classifier_free_guidance
            else prompt_embeds
        )

        if isinstance(token_groups[0][0], int):
            token_groups = [token_groups]

        attention_map_t_plus_one = None
        # 7. Denoising loop
        cross_attention_map_numpy_list, self_attention_map_numpy_list = [], []
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Attend and contrast process
                # print(i,t)
                with torch.enable_grad():
                    if not run_bon:
                        latents = latents.clone().detach().requires_grad_(True)
                        updated_latents = []
                        for j, (latent, token_group, text_embedding) in enumerate(
                            zip(latents, token_groups, text_embeddings)
                        ):
                            # Forward pass of denoising with text conditioning
                            latent = latent.unsqueeze(0)
                            text_embedding = text_embedding.unsqueeze(0)

                            self.unet(latent,t,encoder_hidden_states=text_embedding,cross_attention_kwargs=cross_attention_kwargs,).sample
                            self.unet.zero_grad()

                            # Get max activation value for each subject token
                            cross_attn_map = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"),is_cross = True)
                            self_attn_maps = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"),is_cross=False)

                            # if result_root is not None:
                            #     cross_attention_maps = self.attention_store.aggregate_attention(
                            #         from_where=("down", "mid", "up"), is_cross=True)  # , "down", "mid"
                            #     self_attention_maps = self.attention_store.aggregate_attention(
                            #         from_where=("down", "mid", "up"), is_cross=False)  # , "down", "mid"

                            #     cross_attention_maps_numpy, self_attention_maps_numpy = fn_show_attention_plus(
                            #         cross_attention_maps=cross_attention_maps,
                            #         self_attention_maps=self_attention_maps,
                            #         indices=[2,5],
                            #         K=16,
                            #         attention_res=attn_res[0],
                            #         smooth_attentions=True)  # show average result of top K maps
                            #     cross_attention_map_numpy_list.append(cross_attention_maps_numpy)
                            #     self_attention_map_numpy_list.append(self_attention_maps_numpy)

                            loss = self._compute_contrastive_loss(
                                t=t,
                                cross_attention_maps=cross_attn_map,
                                self_attention_maps=self_attn_maps,
                                attention_maps_t_plus_one=attention_map_t_plus_one,
                                token_groups=token_group,
                                loss_type=loss_fn,
                                temperature=temperature,
                                do_smoothing=do_smoothing,
                                smoothing_kernel_size=smoothing_kernel_size,
                                smoothing_sigma=smoothing_sigma,
                                softmax_normalize=softmax_normalize,
                                softmax_normalize_attention_maps=softmax_normalize_attention_maps,
                            )

                            # If this is an iterative refinement step, verify we have reached the desired threshold for all
                            if i in iterative_refinement_steps:
                                loss, latent = self._perform_iterative_refinement_step(
                                    latents=latent,
                                    token_groups=token_group,
                                    loss=loss,
                                    text_embeddings=text_embedding,
                                    step_size=step_size[i],
                                    t=t,
                                    refinement_steps=refinement_steps,
                                    do_smoothing=do_smoothing,
                                    smoothing_kernel_size=smoothing_kernel_size,
                                    smoothing_sigma=smoothing_sigma,
                                    temperature=temperature,
                                    softmax_normalize=softmax_normalize,
                                    softmax_normalize_attention_maps=softmax_normalize_attention_maps,
                                    attention_maps_t_plus_one=attention_map_t_plus_one,
                                    loss_fn=loss_fn,
                                )

                            # Perform gradient update
                            if i < max_iter_to_alter:
                                if loss != 0:
                                    latent = self._update_latent(
                                        latents=latent,
                                        loss=loss,
                                        step_size=step_size[i],
                                    )

                            updated_latents.append(latent)

                        latents = torch.cat(updated_latents, dim=0)
                    else:
                        losses = []
                        
                        for n_idx, lat_ in enumerate(latents_list):
                            lat_ = lat_.clone().detach().requires_grad_(True)
                            _loss = 0.0
                            for j, (latent, token_group, text_embedding) in enumerate(
                                zip(lat_, token_groups, text_embeddings)
                            ):
                                # Forward pass of denoising with text conditioning
                                latent = latent.unsqueeze(0)
                                text_embedding = text_embedding.unsqueeze(0)

                                self.unet(latent,t,encoder_hidden_states=text_embedding,cross_attention_kwargs=cross_attention_kwargs,).sample
                                self.unet.zero_grad()

                                # Get max activation value for each subject token
                                cross_attn_map = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"),is_cross = True)
                                self_attn_maps = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"),is_cross=False)
                                
                                loss = self._compute_contrastive_loss(
                                    t=t,
                                    cross_attention_maps=cross_attn_map,
                                    self_attention_maps=self_attn_maps,
                                    attention_maps_t_plus_one=attention_map_t_plus_one,
                                    token_groups=token_group,
                                    loss_type=loss_fn,
                                    temperature=temperature,
                                    do_smoothing=do_smoothing,
                                    smoothing_kernel_size=smoothing_kernel_size,
                                    smoothing_sigma=smoothing_sigma,
                                    softmax_normalize=softmax_normalize,
                                    softmax_normalize_attention_maps=softmax_normalize_attention_maps,
                                )

                                _loss += loss.item()
                                
                            losses.append(_loss)
                            
                        best_idx = int(np.argmin(losses))
                        latents = latents_list[best_idx]
                        run_bon = False
                        
                        latents = latents.clone().detach().requires_grad_(True)
                        updated_latents = []
                        for j, (latent, token_group, text_embedding) in enumerate(
                            zip(latents, token_groups, text_embeddings)
                        ):
                            # Forward pass of denoising with text conditioning
                            latent = latent.unsqueeze(0)
                            text_embedding = text_embedding.unsqueeze(0)

                            self.unet(latent,t,encoder_hidden_states=text_embedding,cross_attention_kwargs=cross_attention_kwargs,).sample
                            self.unet.zero_grad()

                            # Get max activation value for each subject token
                            cross_attn_map = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"),is_cross = True)
                            self_attn_maps = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"),is_cross=False)

                            # if result_root is not None:
                            #     cross_attention_maps = self.attention_store.aggregate_attention(
                            #         from_where=("down", "mid", "up"), is_cross=True)  # , "down", "mid"
                            #     self_attention_maps = self.attention_store.aggregate_attention(
                            #         from_where=("down", "mid", "up"), is_cross=False)  # , "down", "mid"

                            #     cross_attention_maps_numpy, self_attention_maps_numpy = fn_show_attention_plus(
                            #         cross_attention_maps=cross_attention_maps,
                            #         self_attention_maps=self_attention_maps,
                            #         indices=[2,5],
                            #         K=16,
                            #         attention_res=attn_res[0],
                            #         smooth_attentions=True)  # show average result of top K maps
                            #     cross_attention_map_numpy_list.append(cross_attention_maps_numpy)
                            #     self_attention_map_numpy_list.append(self_attention_maps_numpy)

                            loss = self._compute_contrastive_loss(
                                t=t,
                                cross_attention_maps=cross_attn_map,
                                self_attention_maps=self_attn_maps,
                                attention_maps_t_plus_one=attention_map_t_plus_one,
                                token_groups=token_group,
                                loss_type=loss_fn,
                                temperature=temperature,
                                do_smoothing=do_smoothing,
                                smoothing_kernel_size=smoothing_kernel_size,
                                smoothing_sigma=smoothing_sigma,
                                softmax_normalize=softmax_normalize,
                                softmax_normalize_attention_maps=softmax_normalize_attention_maps,
                            )

                            # If this is an iterative refinement step, verify we have reached the desired threshold for all
                            if i in iterative_refinement_steps:
                                loss, latent = self._perform_iterative_refinement_step(
                                    latents=latent,
                                    token_groups=token_group,
                                    loss=loss,
                                    text_embeddings=text_embedding,
                                    step_size=step_size[i],
                                    t=t,
                                    refinement_steps=refinement_steps,
                                    do_smoothing=do_smoothing,
                                    smoothing_kernel_size=smoothing_kernel_size,
                                    smoothing_sigma=smoothing_sigma,
                                    temperature=temperature,
                                    softmax_normalize=softmax_normalize,
                                    softmax_normalize_attention_maps=softmax_normalize_attention_maps,
                                    attention_maps_t_plus_one=attention_map_t_plus_one,
                                    loss_fn=loss_fn,
                                )

                            # Perform gradient update
                            if i < max_iter_to_alter:
                                if loss != 0:
                                    latent = self._update_latent(
                                        latents=latent,
                                        loss=loss,
                                        step_size=step_size[i],
                                    )

                            updated_latents.append(latent)

                        latents = torch.cat(updated_latents, dim=0)
                            

                if add_previous_attention_maps and (
                    previous_attention_map_anchor_step is None
                    or i == previous_attention_map_anchor_step
                ):
                    attention_map_t_plus_one = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"),is_cross = True)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # show attention map at each timestep
        # if result_root is not None:
        #     # os.makedirs('{:s}/{:s}'.format(result_root, prompt), exist_ok=True)

        #     # cross_attention_maps_numpy, self_attention_maps_numpy
        #     fig = plt.figure(figsize=(len(cross_attention_map_numpy_list) * 1.5, 6))
        #     rows = 4
        #     columns = len(cross_attention_map_numpy_list)
        #     for j in range(len(cross_attention_map_numpy_list)):
        #         plt.title("t_{}".format(j))
        #         fig.add_subplot(rows, columns, j + 1)
        #         plt.imshow(cross_attention_map_numpy_list[j][0], cmap='viridis')
        #         plt.axis('off')
        #         fig.add_subplot(rows, columns, columns + j + 1)
        #         plt.imshow(cross_attention_map_numpy_list[j][1], cmap='viridis')
        #         plt.axis('off')
        #         fig.add_subplot(rows, columns, 2 * columns + j + 1)
        #         plt.imshow(self_attention_map_numpy_list[j][0], cmap='viridis')
        #         plt.axis('off')
        #         fig.add_subplot(rows, columns, 3 * columns + j + 1)
        #         plt.imshow(self_attention_map_numpy_list[j][1], cmap='viridis')
        #         plt.axis('off')
        #     plt.savefig(f"./{result_root}/{seed}.jpg", bbox_inches='tight', pad_inches=0.2)
        #     plt.close()

        # 8. Post-processing
        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        if not return_dict:
            return (image, has_nsfw_concept)

        return image


class GaussianSmoothing(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)
