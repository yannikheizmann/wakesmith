"""Define fixed audio, feature, language and artefact constants."""

from __future__ import annotations

from dataclasses import dataclass

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280
"""openWakeWord processes 80 ms chunks and emits one embedding frame per chunk."""
FRAME_SECONDS = CHUNK_SAMPLES / SAMPLE_RATE
CONTEXT_FRAMES = 16
"""Embedding frames per classifier window (1.28 s)."""
EMBEDDING_DIM = 96
CLIP_SECONDS = 2.0

MODEL_FILENAME = "{slug}.onnx"
MODEL_CARD_FILENAME = "{slug}.json"
REPORT_FILENAME = "report.md"
ARGS_FILENAME = "args.json"
TRAINING_LOG_FILENAME = "training-log.json"
EVALUATION_FILENAME = "evaluation.json"
TRANSCRIPTS_FILENAME = "transcripts.json"


@dataclass(frozen=True)
class Language:
    """Per-language settings for pronunciation, screening and negative speech."""

    espeak: str
    whisper: str
    kokoro: str | None
    """Kokoro lang code with a native front end; ``None`` uses espeak G2P for the language."""
    mls: str | None
    """Multilingual LibriSpeech subset used as additional negative speech."""


LANGUAGES = {
    "en": Language(espeak="en-us", whisper="en", kokoro="a", mls=None),
    "de": Language(espeak="de", whisper="de", kokoro=None, mls="german"),
    "nl": Language(espeak="nl", whisper="nl", kokoro=None, mls="dutch"),
    "fr": Language(espeak="fr-fr", whisper="fr", kokoro="f", mls="french"),
    "es": Language(espeak="es", whisper="es", kokoro="e", mls="spanish"),
    "it": Language(espeak="it", whisper="it", kokoro="i", mls="italian"),
    "pt": Language(espeak="pt-br", whisper="pt", kokoro="p", mls="portuguese"),
    "pl": Language(espeak="pl", whisper="pl", kokoro=None, mls="polish"),
}

KOKORO_VOICES = (
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
)
KOKORO_EVALUATION_VOICES = ("af_river", "am_puck", "bf_lily", "bm_lewis", "if_sara", "pm_alex")

VOXCPM_TRAIN_VOICES = (
    "A young adult woman, clear mid-pitch voice, moderate pace, calm and professional",
    "A young adult man, warm baritone, steady pace, friendly and articulate",
    "A middle-aged woman, slightly low pitch, measured pace, confident tone",
    "A middle-aged man, deep resonant voice, slow deliberate pace",
    "An older adult woman, soft gentle voice, slower pace, kind tone",
    "An older adult man, gravelly tenor, moderate pace, matter-of-fact",
    "A young woman, bright energetic tone, slightly faster pace",
    "A young man, light tenor, quick conversational pace",
    "A man in his thirties with a noticeable German accent",
    "A woman in her forties with a mild German accent",
    "An older man with a deep voice and a heavy Bavarian accent",
    "A Swiss woman speaking softly with a Swiss accent",
    "An Austrian man, friendly, relaxed Viennese accent",
    "A teenager, casual and slightly bored, light pitch",
    "A child around ten years old, high pitch, excited",
    "A Turkish man in his twenties, conversational, with an accent",
    "A British English accent, clear delivery, moderate pace",
    "A Southern US English accent, warm tone, conversational pace",
    "A French woman with a French accent",
    "A Spanish man with a Spanish accent",
    "An Indian English accent, clear and quick",
    "A Nigerian English accent, confident and warm",
    "A quiet tired voice, close to the microphone, slow and mumbled",
    "A loud voice calling from across a room",
    "A husky voice, low energy, medium-slow pace",
    "A cheerful animated voice, wide pitch range, lively pace",
    "A whispering voice, very soft and breathy",
    "A monotone flat delivery, even speed",
)
VOXCPM_EVALUATION_VOICES = (
    "A middle-aged woman with a Saxon German accent, speaking calmly",
    "A young man from Cologne, upbeat, slightly fast",
    "An elderly woman, slow and careful, slightly shaky voice",
    "A Dutch man with a Dutch accent",
    "A Polish woman with a Polish accent",
    "An Australian man, casual and relaxed",
    "A Scottish woman with a strong Scottish accent",
    "A deep-voiced man speaking from the back of a large room",
    "A girl around twelve years old, bright and impatient",
    "A woman with a cold, nasal and congested voice",
)
