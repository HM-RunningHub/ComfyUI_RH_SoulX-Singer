import uuid
import os
import json
import torchaudio
import folder_paths
from .preprocess.pipeline import PreprocessPipeline

import torch
import json
from tqdm import tqdm
import numpy as np
import soundfile as sf
from collections import OrderedDict
from omegaconf import DictConfig

from .soulxsinger.utils.file_utils import load_config
from .soulxsinger.models.soulxsinger import SoulXSinger
from .soulxsinger.utils.data_processor import DataProcessor
from .rh_lyrics import align_lyrics, clean_metadata, wrap_transcriber

class RunningHub_SoulXSinger_Preprocess_Pipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
            }
        }

    RETURN_TYPES = ('SoulXSinger_Preprocess_Pipeline', )
    RETURN_NAMES = ('SoulX-Singer Preprocess Pipeline', )
    FUNCTION = "load"
    CATEGORY = "RunningHub/SoulX-Singer"

    def load(self, **kwargs):
        pipeline = PreprocessPipeline(
            device="cuda",
            language="Mandarin",
            save_dir=None,
            vocal_sep=True,
            max_merge_duration=60000,
            model_path=os.path.join(folder_paths.models_dir, "Soul-AILab", "SoulX-Singer-Preprocess"),
        )
        return (pipeline, )

class RunningHub_SoulXSinger_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("SoulXSinger_Preprocess_Pipeline", ),
                "audio": ("AUDIO", ),
                "max_merge_duration": ("INT", {"default": 30000, 'min': 10000, 'max': 60000}),
                "language": (["Mandarin", "English", "Cantonese"], {"default": "Mandarin"}),
            },
            "optional": {
                # Force alignment: when the original lyrics are known, Paraformer
                # is biased toward them so the detected text is exactly the
                # reference (timing/pitch still come from the audio). Leave empty
                # for plain ASR.
                "reference_lyrics": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": False}),
                # Clean up ROSVOT artefacts that would otherwise be sung as
                # duplicated syllables / an unnatural pause inside a word.
                "merge_held_notes": ("BOOLEAN", {"default": True,
                                                 "label_on": "ON", "label_off": "OFF"}),
                "merge_repeated_chars": ("BOOLEAN", {"default": True,
                                                     "label_on": "ON", "label_off": "OFF"}),
            },
        }
    
    RETURN_TYPES = ('SoulXSinger_Audio_Metadata', )
    RETURN_NAMES = ('audio metadata', )
    FUNCTION = "process"
    CATEGORY = "RunningHub/SoulX-Singer"

    def save_audio(self, audio, save_path):
        waveform = audio["waveform"]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        sample_rate = int(audio["sample_rate"])
        torchaudio.save(save_path, waveform.cpu(), sample_rate, format="wav")

    def process(self, **kwargs):
        audio = kwargs.get('audio', None)
        pipeline = kwargs.get('pipeline', None)
        max_merge_duration = kwargs.get('max_merge_duration', None)
        language = kwargs.get('language', None)
        reference_lyrics = kwargs.get('reference_lyrics', "") or ""
        merge_held_notes = kwargs.get('merge_held_notes', True)
        merge_repeated_chars = kwargs.get('merge_repeated_chars', True)

        # Attach (or detach) the force-alignment wrapper for this call only.
        # The pipeline object is shared between every Preprocessor node in the
        # graph, so the wrapper is always re-created from the inner transcriber
        # — never left over from a previous node's reference lyrics.
        pipeline.lyric_transcriber = wrap_transcriber(
            pipeline.lyric_transcriber, reference_lyrics
        )

        audio_path = os.path.join(folder_paths.get_temp_directory(), f"{uuid.uuid4()}.wav")
        self.save_audio(audio, audio_path)
        pipeline.max_merge_duration = max_merge_duration
        pipeline.language = language
        pipeline.save_dir = os.path.join(folder_paths.get_temp_directory(), f"{uuid.uuid4()}")
        metadata = pipeline.run(audio_path=audio_path, language=language, max_merge_duration=max_merge_duration)
        metadata = clean_metadata(
            metadata,
            merge_held_notes=merge_held_notes,
            merge_repeated_chars=merge_repeated_chars,
            reference_lyrics=reference_lyrics,
        )
        return (metadata, )

class RunningHub_SoulXSinger_Lyrics_Editor:
    """Replace lyrics in SoulX-Singer metadata while keeping the performance.

    Feed it the metadata produced by ``Preprocessor`` (ideally transcribed with
    ``reference_lyrics`` so the segmentation matches the original words), type
    the new lyrics, and the melody, note durations, note pitches and frame-level
    f0 of the original recording are preserved — only the words change. The
    result feeds ``SVS Processor`` directly.

    ``lyrics preview`` shows how the new lyrics were split across the original
    phrases (``(14/3/1/4/2) 雾散落… / 任由那 / …``), which is the quickest way
    to spot a line that was written with the wrong number of characters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metadata": ("SoulXSinger_Audio_Metadata", ),
                "new_lyrics": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": False}),
            },
            "optional": {
                "preserve_sp": ("BOOLEAN", {"default": True,
                                            "label_on": "keep original phrases", "label_off": "re-flow"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "normalize_digits": ("BOOLEAN", {"default": True}),
                "force_tone4": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("SoulXSinger_Audio_Metadata", "STRING", "STRING")
    RETURN_NAMES = ("audio metadata", "lyrics preview", "warnings")
    FUNCTION = "edit"
    CATEGORY = "RunningHub/SoulX-Singer"

    def edit(self, metadata, new_lyrics, preserve_sp=True, speed=1.0,
             normalize_digits=True, force_tone4=False):
        return align_lyrics(
            metadata,
            new_lyrics,
            preserve_sp=preserve_sp,
            normalize_digits=normalize_digits,
            force_tone4=force_tone4,
            speed=speed,
        )


class RunningHub_SoulXSinger_SVS_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
            }
        }

    RETURN_TYPES = ('SoulXSinger_SVS_Pipeline', )
    RETURN_NAMES = ('SoulX-Singer SVS Pipeline', )
    FUNCTION = "load"
    CATEGORY = "RunningHub/SoulX-Singer"

    def load(self, **kwargs):
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "soulxsinger",
            "config",
            "soulxsinger.yaml",
        )
        config = load_config(config_path)
        device = 'cuda'
        
        model_path = os.path.join(folder_paths.models_dir, "Soul-AILab", "SoulX-Singer", "model.pt")
        model = SoulXSinger(config).to(device)
        print("Model initialized.")
        print("Model parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
        
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)
        if "state_dict" not in checkpoint:
            raise KeyError(
                f"Checkpoint at {model_path} has no 'state_dict' key. "
                "Expected a checkpoint saved with model.state_dict()."
            )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        
        model.eval()
        # model.to(device)
        print("Model checkpoint loaded.")

        phoneset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "soulxsinger",
            "utils",
            "phoneme",
            "phone_set.json",
        )

        data_processor = DataProcessor(
            hop_size=config.audio.hop_size,
            sample_rate=config.audio.sample_rate,
            phoneset_path=phoneset_path,
            device=device,
        )
        return ({'config': config, 'model': model, 'data_processor': data_processor}, )

class RunningHub_SoulXSinger_SVS_Processor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("SoulXSinger_SVS_Pipeline", ),
                "control": (["melody", "score"], {"default": "melody"}),
                "prompt_wav": ("AUDIO", ),
                "prompt_metadata": ("SoulXSinger_Audio_Metadata", ),
                # "prompt_metadata": ("STRING", {"default": "", "multiline": True}),
                "target_metadata": ("SoulXSinger_Audio_Metadata", ),
                "seed": ("INT", {"default": 12306, "min": 0, "max": 4294967295}),
                # "save_dir": ("STRING", {"default": ""}),
                # "auto_shift": ("BOOLEAN", {"default": False}),
                # "pitch_shift": ("INT", {"default": 0, "min": -12, "max": 12}),
            }
        }
    
    RETURN_TYPES = ('AUDIO', )
    RETURN_NAMES = ('audio', )
    FUNCTION = "process"
    CATEGORY = "RunningHub/SoulX-Singer"

    def save_audio(self, audio, save_path):
        waveform = audio["waveform"]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        sample_rate = int(audio["sample_rate"])
        torchaudio.save(save_path, waveform.cpu(), sample_rate, format="wav")
    
    def process(self, **kwargs):
        pipeline = kwargs.get('pipeline', None)
        config = pipeline.get('config', None)
        model = pipeline.get('model', None)
        data_processor = pipeline.get('data_processor', None)
        prompt_wav = kwargs.get('prompt_wav', None)
        control = kwargs.get('control', None)

        prompt_wav_path = os.path.join(folder_paths.get_temp_directory(), f"prompt_{uuid.uuid4()}.wav")
        self.save_audio(prompt_wav, prompt_wav_path)

        device = 'cuda'
        # save_dir = os.path.join(folder_paths.get_temp_directory(), f"svs_{uuid.uuid4()}")
        auto_shift = True
        pitch_shift = 0

        prompt_metadata = kwargs.get('prompt_metadata', None)
        target_metadata = kwargs.get('target_metadata', None)

        # os.makedirs(save_dir, exist_ok=True)

        # with open(args.prompt_metadata_path, "r", encoding="utf-8") as f:
        # prompt_meta_list = json.load(prompt_metadata)
        prompt_meta_list = json.loads(prompt_metadata)
        if not prompt_meta_list:
            raise ValueError("Prompt metadata is empty. Please run preprocess on prompt audio first.")
        prompt_meta = prompt_meta_list[0]  # load the first segment as the prompt

        # # with open(args.target_metadata_path, "r", encoding="utf-8") as f:
        # target_meta_list = json.load(f)
        target_meta_list = json.loads(target_metadata)
        infer_prompt_data = data_processor.process(prompt_meta, prompt_wav_path)

        # assert len(target_meta_list) > 0, "No target segments found in the target metadata."
        generated_len = int(target_meta_list[-1]["time"][1] / 1000 * config.audio.sample_rate)
        generated_merged = np.zeros(generated_len, dtype=np.float32)

        for idx, target_meta in enumerate(
            tqdm(target_meta_list, total=len(target_meta_list), desc="Inferring segments"),
        ):
            start_sample_idx = int(target_meta["time"][0] / 1000 * config.audio.sample_rate)
            end_sample_idx = int(target_meta["time"][1] / 1000 * config.audio.sample_rate)
            infer_target_data = data_processor.process(target_meta, None)

            infer_data = {
                "prompt": infer_prompt_data,
                "target": infer_target_data,
            }

            with torch.no_grad():
                model.to(device)
                generated_audio = model.infer(
                    infer_data,
                    auto_shift=auto_shift,
                    pitch_shift=pitch_shift,
                    n_steps=config.infer.n_steps,
                    cfg=config.infer.cfg,
                    control=control,
                )
                model.to('cpu')

            generated_audio = generated_audio.squeeze().cpu().numpy()
            generated_merged[start_sample_idx : start_sample_idx + generated_audio.shape[0]] = generated_audio

        # merged_path = os.path.join(save_dir, "generated.wav")
        # sf.write(merged_path, generated_merged, 24000)
        wave = torch.from_numpy(generated_merged)
        
        if wave.dim() == 1:      
            wave = wave.unsqueeze(0)      
        wave = wave.unsqueeze(0)

        audio_obj = {
            "waveform": wave,
            "sample_rate": 24000,
        }
        return (audio_obj, )

NODE_CLASS_MAPPINGS = {
    "RunningHub SoulX-Singer Preprocess Pipeline": RunningHub_SoulXSinger_Preprocess_Pipeline,
    "RunningHub SoulX-Singer Preprocessor": RunningHub_SoulXSinger_Preprocessor,
    "RunningHub SoulX-Singer Lyrics Editor": RunningHub_SoulXSinger_Lyrics_Editor,
    "RunningHub SoulX-Singer SVS Loader": RunningHub_SoulXSinger_SVS_Loader,
    "RunningHub SoulX-Singer SVS Processor": RunningHub_SoulXSinger_SVS_Processor,
}