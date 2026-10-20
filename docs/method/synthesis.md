# Synthesis and screening

## Engines

**Kokoro-82M** provides most positives. Each training speaker is a random convex blend of two
or three of its 48 training voices, spoken at 0.75–1.25× speed. English runs mix the American
and British front ends with Spanish, French, Italian, Portuguese and Hindi front ends, which
produce accented English; other languages use espeak-ng phonemes for that language.

**VoxCPM2** generates speakers from free-text descriptions ("an older man with a heavy
Bavarian accent", "a whispering voice"), adding speaking styles that blended voices lack. It is
the slowest engine and runs in a separate process that returns its memory when done.

**Piper LibriTTS** (904 speakers, English only) supplies confusable negatives. livekit-wakeword's
wrapper is used with two replacements: phonemes come from the bundled espeak-ng library, and
only leading and trailing silence is trimmed (the upstream wrapper removes silent frames
inside clips, which cuts stop closures and weak onsets).

## Confusable phrases

Adversarial negatives combine three sources:

- **Phonetic neighbours**: words and two-word phrases from the language's 60,000 most frequent
  words whose espeak pronunciation lies close to the wake word's (`tom chum`, `tom autumn`,
  `tatum` for "tomtom").
- **Fragments**: dropped words of a phrase and the leading or trailing half of a long word.
- **Your phrases** from `--negative`.

Fragments and your phrases are repeated to make up about a third of adversarial clips.
Neighbours that the screen would accept as the wake word (inflections such as "computers")
are removed, so no phrase is trained as both.

## The screen

Text-to-speech engines often mispronounce invented words; in development, Whisper recognised
fewer than one in ten of Piper's isolated clips of an invented two-syllable word. Every positive is therefore
transcribed by Whisper large-v3-turbo in the wake word's language and accepted only if the
phonetic matcher agrees:

- the transcript, minus leading fillers ("hey", "okay"), is converted to espeak IPA;
- its **consonant skeleton must equal** the wake word's, allowing a trailing *s*;
- its **vowels may differ** by up to half of their normalised edit distance.

Vowel quality varies across voices and accents; a changed, missing or extra consonant makes
a different word (`tomcom`, `bomtom`, `tom tom tom` for "tomtom"). Adversarial clips that the matcher accepts are
dropped. Accepted counts per split are written to `clips/screen.json` and the report.
