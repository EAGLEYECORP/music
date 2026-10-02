# studiomix

Give it a **lead vocal**, optional **ad-libs**, and an **instrumental**. You get back a tuned, mixed
and mastered song that is ready to upload to Spotify, Apple Music, YouTube, TikTok, or any
distributor (DistroKid, TuneCore, CD Baby…). It runs on a computer or on an Android phone (Termux).

```bash
studiomix vocal.wav beat.wav --adlibs adlibs.wav --preset trap --name "My Song"
```

```
out/
  My Song_master_24bit_48k.wav        <- upload this to your distributor
  My Song_master_16bit_44.1k.wav      <- CD-quality / for distributors that require 16-bit
  My Song_master_preview_320k.mp3     <- for sharing with people (don't upload it to stores)
  My Song_premaster_mix_24bit.wav     <- the mix before mastering, peaks at -6 dBFS (for a mastering engineer)
  My Song_vocal_stem_24bit.wav        <- processed lead vocal on its own
  My Song_adlib_stem_24bit.wav        <- processed ad-libs on their own
  My Song_instrumental_stem_24bit.wav <- processed beat on its own
  My Song_report.txt / .json          <- loudness, peaks, what was done, delivery checks
```

## Install

Requires Python 3.10+. The only dependencies are numpy, scipy and pyloudnorm.

```bash
pip install -e .              # minimal: works everywhere
pip install -e ".[fast]"      # optional: numba JIT for the dynamics loops (desktop only)
pip install -e ".[dev]"       # to run the tests
```

WAV files always load. For `mp3`, `m4a`, `flac` and other formats, and to get the MP3 preview,
install `ffmpeg`.

### Android phone (Termux)

1. Install **Termux from F-Droid**. The Play Store version is outdated and broken.
2. In Termux:

```bash
pkg install git
git clone https://github.com/eagleyecorp/music && cd music
bash scripts/termux-setup.sh
```

The script installs Python, numpy, scipy and ffmpeg as ready-made Termux packages, so nothing has
to compile. It then installs studiomix and asks for access to your phone's storage. After that:

```bash
cd ~/storage/downloads
studiomix vocal.wav beat.wav -a adlibs.wav -p trap -o master
```

The finished files appear in your phone's **Downloads/master** folder. A phone is slower than a
computer. Expect a few minutes per song, and keep Termux open while it runs.

## What it does

**Auto-tune (pitch correction)**
- **Finds the key automatically.** It reads the harmony of the beat (drums are filtered out first) and checks how well the singer's melody fits each candidate scale. If it isn't confident, it falls back to safe chromatic tuning and says so in the report. You can always set the key yourself with `--key "F# minor"`.
- **Tracks pitch robustly.** It uses probabilistic YIN (pYIN): every candidate pitch is weighed under a whole range of thresholds, then the most consistent path through time is chosen (Viterbi). This all but eliminates the octave jumps that make cheap auto-tune glitch. The analysis looks only below 1.2 kHz, so breath, rasp and "s" sounds don't confuse it, and loud frames count as evidence of singing, so breathy notes still get tuned.
- **Decides notes using the whole phrase.** Because it works on the finished recording rather than live, it can look ahead: the sustained part of a note decides which note it is, so a scoop into a note or a fall at its end can't snap it to the wrong one. A real-time plug-in can't do this.
- **Sub-sample accurate.** New pitch periods use the singer's exact (fractional) period and are placed with sub-sample precision. High voices land within about a cent instead of being off by a rounding error.
- **Retune speed:** `0` ms is the instant, robotic snap (T-Pain / trap). 20–40 ms is modern pop. 80 ms and up is natural.
- **Humanize:** keeps vibrato and expression on held notes while the note centre still lands in tune.
- **Keeps the voice's natural tone.** It re-spaces the voice's own pitch cycles (PSOLA) instead of speeding the audio up or down, so there's no chipmunk effect. Parts that don't need correcting pass through bit-for-bit unchanged.
- **Styles:** `--tune hard | pop | natural | off`
- Scales: major, minor, harmonic-minor, major/minor-pentatonic, blues, chromatic

**Measured, not guessed.** `python tools/bench_tune.py` renders synthetic singers whose true pitch
is known: bass to C6, vibrato, scoops, drift, breathy and near-whisper delivery, and "s"/"sh"
consonants. It then scores the tuner against a perfectly tuned render of the same performance.

| (7 voices, hard tune)                 | first version | now |
|---------------------------------------|------|------|
| notes landing in tune (< 10 cents)    | 46%  | **95%** |
| average note error                    | 328 cents | **1.25 cents** |
| tracking errors (octave jumps etc.)   | 26%  | **0.3%** |
| sung audio left untuned (missed)      | 40%  | **5%** |
| breathy voice: notes correct          | 7%   | **100%** |
| consonants wrongly treated as pitched | 5.8% | 7.6% |

**On real voices** (`python tools/eval_real_voices.py`, which fetches CC-BY LibriSpeech speech
and a solo trumpet):

- **Tracker:** compared with frames where librosa's pYIN and Praat agree with each other, it is
  within 50 cents on **99.3–99.9%** of frames. It catches 95–99% of the voiced audio and marks at
  most 1.1% of unvoiced audio as pitched.
- **Real voice sung out of tune, then hard-tuned** (measured with Praat, not our own tracker):
  notes within 10 cents rose from 27–50% to **67–100%**. The deep voice (about 78 Hz) is the
  weakest case. Speech moves pitch faster within a syllable than sustained singing does.

Known limits: near-whispered singing is still tracked only about 75% of the time. A singer more
than about 45 cents off sits halfway between two notes, so it snaps to whichever scale note is
nearer, which may not be the one they meant. Pass `--key` to rule out non-scale notes.

**Lead vocal and ad-libs**
1. Folds a stereo vocal to mono, then sets a fixed working level so every later stage behaves the same on every take
2. 4th-order high-pass filter to remove rumble, plosives and mic-handling noise
3. Pitch correction, done on the clean voice before any compression or saturation
4. Soft gate (downward expander) to clean up room noise between phrases
5. **Adaptive EQ**: measures the vocal's spectrum and corrects it toward a balanced lead-vocal curve. It cuts up to 6 dB but boosts at most 3 dB.
6. Mud cut (300 Hz) and boxiness cut (800 Hz), then presence (3.5 kHz) and air (12 kHz) boosts
7. **Two-stage compression**: a fast peak catcher feeding a slow, smooth leveler
8. Split-band **de-esser** with a threshold that adapts to the singer
9. Parallel tape-style saturation, oversampled, for warmth
10. **Vocal rider** (lead only): automatic fader moves that keep the vocal at the same level relative to the beat, so it doesn't get buried in loud choruses
11. Tempo-synced ping-pong delay and a plate-style reverb. Both are ducked by the dry vocal so the words stay clear.

**Ad-libs** go through the same chain, but thinner (high-pass at 150 Hz), with more compression,
presence, delay and reverb, about 4 dB under the lead. **Each ad-lib phrase alternates
left/right**, the classic hip-hop spread. You can pass several files with `-a file1 file2`.

**Instrumental**
- Subsonic filter
- **Vocal-keyed dynamic EQ**: the beat dips a few dB in the 1.5–5 kHz range only while the vocals are singing. This makes space for them without turning the beat down.

**Master bus**
1. Tonal balance: either matches a **reference song** (`--reference`) or nudges the mix toward the spectral tilt of commercial releases
2. Phase-coherent 3-band compression (Linkwitz-Riley crossovers)
3. Bus "glue" compressor
4. Mono low end below 120 Hz, slightly wider highs
5. Oversampled soft clipper, then a look-ahead **true-peak limiter**. Loudness is adjusted repeatedly until the integrated loudness hits the target within ±0.1 LU.
6. The 44.1 kHz master gets its own limiting pass (instead of resampling an already-limited file), then TPDF dither for the 16-bit export

All loudness readings are ITU-R BS.1770-4 / EBU R128 LUFS. All peak readings are 4×-oversampled true peak.

## Presets

| preset      | sound                                                     | auto-tune          | loudness  |
|-------------|-----------------------------------------------------------|--------------------|-----------|
| `pop`       | bright, upfront vocal, polished (default)                 | pop (30 ms)        | -11 LUFS  |
| `hiphop`    | dry, loud, in-your-face vocal, heavy mono low end         | tight (10 ms)      |  -9 LUFS  |
| `trap`      | hard robotic tune, wide ad-libs, very loud                | hard (0 ms)        | -8.5 LUFS |
| `rnb`       | warm vocal, lush reverb, wider image                      | smooth (30 ms)     | -11 LUFS  |
| `rock`      | vocal sits inside dense guitars, more bite                | natural (70 ms)    | -10 LUFS  |
| `acoustic`  | natural and dynamic, light compression                    | natural (90 ms)    | -14 LUFS  |
| `streaming` | exactly the Spotify/YouTube reference level, max dynamics | pop (25 ms)        | -14 LUFS  |

**About loudness and Spotify:** Spotify, YouTube, Apple Music and Tidal turn every song to roughly
-14 LUFS. A louder master doesn't play louder. It gets turned down and keeps its denser, punchier
character. Following Spotify's own guidance, any master louder than -14 LUFS is automatically
limited to **-2 dBTP** instead of -1 dBTP, so it doesn't distort when encoded to Ogg/AAC. You can
override this with `--ceiling`.

## Fine tuning

```bash
studiomix vox.wav beat.mp3 -p hiphop -a adlibs.wav hype.wav --tune hard --key "C# minor" --adlib-level -3
```

| option | what it does |
|---|---|
| `-a FILE [FILE ...]` | ad-lib tracks |
| `--tune hard\|pop\|natural\|off` | auto-tune style |
| `--key "C# minor"` | skip key detection (also `Bbm`, `G minor-pentatonic`, `chromatic`…) |
| `--retune-ms 15` | custom retune speed (0 = robotic) |
| `--humanize 0.5` | keep half the vibrato on long notes |
| `--tune-amount 0.7` | partial correction |
| `--adlib-level -3` | ad-libs vs. the lead (dB) |
| `--adlib-pan 0.8` | ad-lib left/right spread (0-1) |
| `--vocal-level 1.5` | lead vocal vs. the beat (dB, + = louder) |
| `--reverb 0.2` / `--delay 0.1` | vocal reverb / delay amount (0-1) |
| `--air 4` / `--presence 3` | vocal brightness / bite (dB) |
| `--vocal-comp 1.4` | vocal compression amount (0-2) |
| `--deess 10` | max de-essing (dB, 0 = off) |
| `--carve 4` | how much the beat makes room for the vocal (dB) |
| `--lufs -10` / `--ceiling -1` | loudness target / true-peak ceiling |
| `--reference song.wav` | match the tone of a song you like |
| `--offset-ms 25` | shift the vocals if they were exported late/early |

Run `studiomix --help` for everything.

## Tips for the best result

- Export the lead, the ad-libs and the beat **from the same start point** (bar 1), as WAV, with no effects, auto-tune or limiter on the vocals
- If the key detection is wrong (the report shows the key it used), pass `--key`. Hard tune in the wrong key sounds bad.
- Leave headroom: no clipping in either file. Anything that peaks below about -3 dBFS is fine.
- If the beat is already mastered and loud, that's OK. The chain re-balances it.
- Try `--reference` with a professionally released song in the same genre for the closest match to "that sound"

## Try it without your own files

```bash
python tools/make_demo.py demo/      # an out-of-tune synthetic singer, ad-libs and a beat in A minor
studiomix demo/demo_vocal.wav demo/demo_beat.wav -a demo/demo_adlibs.wav -p trap
```

## Tests

```bash
pytest
```

The tests also run a full song with numba, pedalboard and soundfile blocked from importing, which
is exactly the situation on Termux.
