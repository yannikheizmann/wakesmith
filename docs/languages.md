# Languages

`--language` decides how the wake word is spoken, screened and contrasted:

| Component | Effect of `--language` |
|---|---|
| Pronunciation | espeak-ng voice used for phonemes, confusables and the screen |
| Kokoro | native front end (`en`, `fr`, `es`, `it`, `pt`) or espeak-ng phonemes for the language |
| VoxCPM2 | text spoken in the language (VoxCPM2 covers about 30 languages) |
| Piper | used only for English |
| Whisper screen | transcription language |
| Confusables | the language's 60,000 most frequent words (`wordfreq`) |
| Negative speech | Multilingual LibriSpeech in the language, added to English ACAV100M |

| Code | Language | espeak-ng | Negative speech |
|---|---|---|---|
| `en` | English | `en-us` | ACAV100M |
| `de` | German | `de` | + MLS German |
| `nl` | Dutch | `nl` | + MLS Dutch |
| `fr` | French | `fr-fr` | + MLS French |
| `es` | Spanish | `es` | + MLS Spanish |
| `it` | Italian | `it` | + MLS Italian |
| `pt` | Portuguese | `pt-br` | + MLS Portuguese |
| `pl` | Polish | `pl` | + MLS Polish |

## Mixed-language households

A wake word is usually spoken in one language, but surrounded by speech in others. Train in the
language of the word and add the household's language as negative speech:

```bash
uv run wakesmith "nova" --language en --negatives de
```

## Language independence

The feature extractor is acoustic: Google's speech embedding model was trained mostly on
English but represents sounds, not words, so invented names work in any language. What a
language changes is how the synthetic voices say the word and which words it must not be
confused with.
