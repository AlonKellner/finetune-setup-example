"""A custom Wav2Vec2 feature extractor to add custom padding logic."""

from typing import Any

import numpy as np
import torch
from transformers import (
    BatchFeature,
    SeamlessM4TFeatureExtractor,
    Wav2Vec2FeatureExtractor,
)
from transformers.utils import (
    PaddingStrategy,
    TensorType,
    is_tf_tensor,
    is_torch_available,
    is_torch_tensor,
    logging,
    to_numpy,
)

from .audio_utils import (
    spectrogram,
    spectrogram_batch,
)

logger = logging.get_logger(__name__)


class CustomFeatureExtractorMixin:
    """Custom Wav2Vec2 feature extractor to add custom padding logic."""

    def __init__(
        self, max_batch_length: int | None = None, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_batch_length = max_batch_length

    def pad(
        self,
        processed_features: BatchFeature
        | list[BatchFeature]
        | dict[str, BatchFeature]
        | dict[str, list[BatchFeature]]
        | list[dict[str, BatchFeature]],
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = None,
        return_tensors: str | TensorType | None = None,
    ) -> BatchFeature:
        """Pad features.

        Pad input values / input vectors or a batch of input values / input vectors up to predefined length or to the
        max sequence length in the batch.

        Padding side (left/right) padding values are defined at the feature extractor level (with `self.padding_side`,
        `self.padding_value`)

        <Tip>

        If the `processed_features` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
        result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
        PyTorch tensors, you will lose the specific device of your tensors however.

        </Tip>

        Args:
            processed_features ([`BatchFeature`], list of [`BatchFeature`], `Dict[str, List[float]]`, `Dict[str, List[List[float]]` or `List[Dict[str, List[float]]]`):
                Processed inputs. Can represent one input ([`BatchFeature`] or `Dict[str, List[float]]`) or a batch of
                input values / vectors (list of [`BatchFeature`], *Dict[str, List[List[float]]]* or *List[Dict[str,
                List[float]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.

                Instead of `List[float]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors),
                see the note above for the return type.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(processed_features, list | tuple) and isinstance(
            processed_features[0],  # type: ignore
            dict | BatchFeature,
        ):
            processed_features = {
                key: [example[key] for example in processed_features]
                for key in processed_features[0]  # type: ignore
            }

        # The model's main input name, usually `input_values`, has be passed for padding
        if self.model_input_names[0] not in processed_features:
            raise ValueError(
                "You should supply an instance of `transformers.BatchFeature` or list of `transformers.BatchFeature`"
                f" to this method that includes {self.model_input_names[0]}, but you provided"
                f" {list(processed_features.keys())}"  # type: ignore
            )

        required_input = processed_features[self.model_input_names[0]]  # type: ignore
        return_attention_mask = (
            return_attention_mask
            if return_attention_mask is not None
            else self.return_attention_mask
        )

        if len(required_input) == 0:
            if return_attention_mask:
                processed_features["attention_mask"] = []  # type: ignore
            return processed_features  # type: ignore

        # If we have PyTorch/TF tensors or lists as inputs, we cast them as Numpy arrays
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = self._get_first_element(required_input)  # type: ignore

        if return_tensors is None:
            return_tensors = self._infer_tensor_type(first_element)

        for key, value in processed_features.items():  # type: ignore
            if isinstance(value[0], int | float):  # type: ignore
                processed_features[key] = to_numpy(value)
            else:
                processed_features[key] = [to_numpy(v) for v in value]  # type: ignore

        # Convert padding_strategy in PaddingStrategy
        padding_strategy = self._get_padding_strategies(
            padding=padding,  # type: ignore
            max_length=max_length,
        )

        required_input = processed_features[self.model_input_names[0]]  # type: ignore

        batch_size = self._get_batch_size(processed_features, required_input)  # type: ignore

        truncated_inputs = self._truncate_inputs(
            processed_features,  # type: ignore
            max_length,
            truncation,
            pad_to_multiple_of,
            batch_size,  # type: ignore
        )

        if (self.max_batch_length is not None) and (
            padding_strategy != PaddingStrategy.DO_NOT_PAD
        ):
            max_length = self.max_batch_length // batch_size
            padding_strategy = PaddingStrategy.MAX_LENGTH
        elif padding_strategy == PaddingStrategy.LONGEST:
            # make sure that `max_length` cannot be longer than the longest truncated length
            max_length = max(
                len(input_slice[self.model_input_names[0]])
                for input_slice in truncated_inputs
            )
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = self._get_padded_outputs(
            max_length,
            pad_to_multiple_of,
            return_attention_mask,
            padding_strategy,
            batch_size,
            truncated_inputs,
        )

        return BatchFeature(batch_outputs, tensor_type=return_tensors)

    def _get_first_element(
        self, required_input: list[float | int | np.ndarray]
    ) -> float | int | np.ndarray:
        first_element = required_input[0]  # type: ignore
        if isinstance(first_element, list | tuple):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            index = 0
            while len(required_input[index]) == 0:  # type: ignore
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]  # type: ignore
        return first_element

    def _get_batch_size(
        self, processed_features: dict[str, Any], required_input: list[Any]
    ) -> int:
        batch_size = len(required_input)
        if not all(len(v) == batch_size for v in processed_features.values()):
            raise ValueError(
                "Some items in the output dictionary have a different batch size than others."
            )

        return batch_size

    def _get_padded_outputs(
        self,
        max_length: int | None,
        pad_to_multiple_of: int | None,
        return_attention_mask: bool,
        padding_strategy: PaddingStrategy,
        batch_size: int,
        truncated_inputs: list[dict],
    ) -> dict:
        batch_outputs = {}
        for i in range(batch_size):
            # padding
            outputs = self._pad(
                truncated_inputs[i],
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                if value.dtype is np.dtype(np.float64):
                    _value = value.astype(np.float32)
                else:
                    _value = value
                batch_outputs[key].append(_value)
        return batch_outputs

    def _truncate_inputs(
        self,
        processed_features: dict[str, Any],
        max_length: int | None,
        truncation: bool,
        pad_to_multiple_of: int | None,
        batch_size: int,
    ) -> list[dict]:
        truncated_inputs = []
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in processed_features.items()}  # type: ignore
            # truncation
            inputs_slice = self._truncate(
                inputs,  # type: ignore
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                truncation=truncation,
            )
            truncated_inputs.append(inputs_slice)
        return truncated_inputs

    def _infer_tensor_type(
        self, first_element: int | float | list | tuple | np.ndarray | Any
    ) -> str:
        if is_tf_tensor(first_element):
            return_tensors = "tf"
        elif is_torch_tensor(first_element):
            return_tensors = "pt"
        elif isinstance(first_element, int | float | list | tuple | np.ndarray):
            return_tensors = "np"
        else:
            raise ValueError(
                f"type of {first_element} unknown: {type(first_element)}. "
                "Should be one of a python, numpy, pytorch or tensorflow object."
            )

        return return_tensors

    def _pad(
        self,
        processed_features: dict[str, np.ndarray] | BatchFeature,
        max_length: int | None = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = None,
    ) -> dict:
        """
        Pad inputs (on left/right and up to predefined length or max length in the batch).

        Args:
            processed_features (`Union[Dict[str, np.ndarray], BatchFeature]`):
                Dictionary of input values (`np.ndarray[float]`) / input vectors (`List[np.ndarray[float]]`) or batch
                of inputs values (`List[np.ndarray[int]]`) / input vectors (`List[np.ndarray[int]]`)
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see below)
            padding_strategy (`PaddingStrategy`, *optional*, default to `PaddingStrategy.DO_NOT_PAD`):
                PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The feature_extractor padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
                    - 'random': pads randomly on both sides of the sequences
            pad_to_multiple_of (`int`, *optional*):
                Integer if set will pad the sequence to a multiple of the provided value. This is especially useful to
                enable the use of Tensor Core on NVIDIA hardware with compute capability `>= 7.5` (Volta), or on TPUs
                which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Set to False to avoid returning attention mask (default: set to model specifics)
        """
        required_input = processed_features[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = (
            padding_strategy != PaddingStrategy.DO_NOT_PAD
            and (max_length is not None)
            and (len(required_input) < max_length)
        )

        if return_attention_mask and "attention_mask" not in processed_features:
            processed_features["attention_mask"] = np.ones(
                len(required_input), dtype=np.int32
            )

        if needs_to_be_padded:
            assert max_length is not None, (
                "max_length should not be None when padding is required."
            )
            difference = max_length - len(required_input)
            if self.padding_side == "right":
                if return_attention_mask:
                    processed_features["attention_mask"] = np.pad(
                        processed_features["attention_mask"], (0, difference)
                    )
                padding_shape = (
                    ((0, difference), (0, 0))
                    if self.feature_size > 1
                    else (0, difference)
                )
                processed_features[self.model_input_names[0]] = np.pad(
                    required_input,
                    padding_shape,
                    "constant",
                    constant_values=self.padding_value,
                )
            elif self.padding_side == "left":
                if return_attention_mask:
                    processed_features["attention_mask"] = np.pad(
                        processed_features["attention_mask"], (difference, 0)
                    )
                padding_shape = (
                    ((difference, 0), (0, 0))
                    if self.feature_size > 1
                    else (difference, 0)
                )
                processed_features[self.model_input_names[0]] = np.pad(
                    required_input,
                    padding_shape,
                    "constant",
                    constant_values=self.padding_value,
                )
            elif self.padding_side == "random":
                right_pad = np.random.randint(0, difference + 1)
                left_pad = difference - right_pad
                if return_attention_mask:
                    processed_features["attention_mask"] = np.pad(
                        processed_features["attention_mask"], (left_pad, right_pad)
                    )
                padding_shape = (
                    ((left_pad, right_pad), (0, 0))
                    if self.feature_size > 1
                    else (left_pad, right_pad)
                )
                processed_features[self.model_input_names[0]] = np.pad(
                    required_input,
                    padding_shape,
                    "constant",
                    constant_values=self.padding_value,
                )
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return processed_features  # type: ignore


class CustomWav2Vec2FeatureExtractor(
    CustomFeatureExtractorMixin, Wav2Vec2FeatureExtractor
):
    """Wav2Vec2 feature extracture with a custom mixin."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class CustomSeamlessM4TFeatureExtractor(
    CustomFeatureExtractorMixin, SeamlessM4TFeatureExtractor
):
    """Wav2Vec2 feature extracture with a custom mixin."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __call__(  # noqa: PLR0912
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str | PaddingStrategy = True,
        pad_to_multiple_of: int | None = 2,
        max_length: int | None = None,
        truncation: bool = False,
        return_tensors: str | TensorType | None = None,
        sampling_rate: int | None = None,
        return_attention_mask: bool | None = None,
        do_normalize_per_mel_bins: bool | None = True,
        **_: Any,
    ) -> BatchFeature | list[BatchFeature]:
        """
        Featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `torch.Tensor`, `list[float]`, `list[np.ndarray]`, `list[torch.Tensor]`,
            `list[list[float]]`, `list[list[list[float]]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array,
                a torch tensor, a list of float values, a list of numpy arrays, a list of torch tensors,
                a list of list of float values or a list of a list of list of float values.
                If `raw_speech` is a one-dimensional `np.ndarray`, `torch.Tensor` or a `list[float]`, `raw_speech` is
                considered a single-channel, single-sample sound. In all other cases, the first dimension of
                `raw_speech`, whether from an `np.ndarray`, a `torch.Tensor` or a `list[...]`,
                corresponds to the number of samples in the batch, and the number of channels
                (i.e. mono or stereo character) is derived from the other dimensions
                (1D -> single-channel waveform batches; 2D-> stereo-channel waveform batches).
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            pad_to_multiple_of (`int`, *optional*, defaults to 2):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>

                For SeamlessM4T models, `attention_mask` should always be passed for batched inference, to avoid subtle
                bugs.

                </Tip>

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            do_normalize_per_mel_bins (`bool`, *optional*, defaults to `True`):
                Whether or not to zero-mean unit-variance normalize the input per mel-channel.
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to the tokenizer or the feature
                extractor.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        return_attention_mask = (
            return_attention_mask
            if return_attention_mask is not None
            else self.return_attention_mask
        )

        is_batched_numpy = (
            isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        )
        if is_batched_numpy and len(raw_speech.shape) > 3:
            raise ValueError(
                f"Only mono-channel or stereo-channel audio is supported for input to {self}"
            )

        acceptable_types = (
            (torch.Tensor, np.ndarray, tuple, list)
            if is_torch_available()
            else (np.ndarray, tuple, list)
        )
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, list | tuple)
            and (isinstance(raw_speech[0], acceptable_types))
        )

        if is_batched:
            raw_speech = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(
            np.float64
        ):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        # extract fbank features
        _features = self._extract_fbank_features(raw_speech)

        if not is_batched:
            _features = [_features]

        all_padded_inputs = []
        for feat in _features:
            features = feat
            if do_normalize_per_mel_bins:
                # torch defaults to ddof=1, and numpy defaults to ddof=0
                features = (features - features.mean(0, keepdims=True)) / (
                    features.std(0, ddof=1, keepdims=True) + 1e-7
                )

            # convert into correct format for padding
            encoded_inputs = BatchFeature({"input_features": features})

            padded_inputs = self.pad(
                encoded_inputs,
                padding=padding,
                max_length=max_length,
                truncation=truncation,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="np",
            )

            # SeamlessM4T needs to process extracted features
            input_features = padded_inputs.get("input_features")
            attention_mask = padded_inputs.pop("attention_mask")

            if len(input_features.shape) == 2:
                input_features = np.expand_dims(input_features, axis=0)
                attention_mask = np.expand_dims(attention_mask, axis=0)

            batch_size, num_frames, num_channels = input_features.shape

            remainder = num_frames % self.stride
            if remainder != 0:
                input_features = input_features[:, : num_frames - remainder, :]
                attention_mask = attention_mask[:, : num_frames - remainder]

            input_features = np.reshape(
                input_features,
                (batch_size, num_frames // self.stride, num_channels * self.stride),
            )

            indices = np.arange(0, num_frames - remainder)
            attention_mask = attention_mask[:, indices % self.stride == 1]

            padded_inputs["input_features"] = input_features
            if return_attention_mask:
                padded_inputs["attention_mask"] = attention_mask

            if return_tensors is not None:
                padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

            all_padded_inputs.append(padded_inputs)
        return all_padded_inputs

    def _extract_fbank_features(
        self,
        waveform: np.ndarray | list[np.ndarray],
    ) -> np.ndarray | list[np.ndarray]:
        """
        Get mel-filter bank features using TorchAudio.

        Note that TorchAudio requires 16-bit signed integers as inputs
        and hence the waveform should not be normalized before feature extraction.
        """
        if isinstance(waveform, list):
            waveform = [(w[0] if len(w.shape) == 2 else w) for w in waveform]
            waveform = [
                np.squeeze(w) * (2**15) for w in waveform
            ]  # Kaldi compliance: 16-bit signed integers
            features = spectrogram_batch(
                waveform,
                self.window,
                frame_length=400,
                hop_length=160,
                fft_length=512,
                power=2.0,
                center=False,
                preemphasis=0.97,
                mel_filters=self.mel_filters,
                log_mel="log",
                mel_floor=1.192092955078125e-07,
                remove_dc_offset=True,
            )
            features = [f.T for f in features]
            return features
        else:
            # by default, it extracts the left channel if stereo
            if len(waveform.shape) == 2:
                waveform = waveform[0]

            waveform = np.squeeze(waveform) * (
                2**15
            )  # Kaldi compliance: 16-bit signed integers
            features = spectrogram(
                waveform,
                self.window,
                frame_length=400,
                hop_length=160,
                fft_length=512,
                power=2.0,
                center=False,
                preemphasis=0.97,
                mel_filters=self.mel_filters,
                log_mel="log",
                mel_floor=1.192092955078125e-07,
                remove_dc_offset=True,
            ).T
            return features
