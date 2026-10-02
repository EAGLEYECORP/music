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

## Already have a mix? Master it

```bash
studiomix master rough_mix.wav -p hiphop --vocal-lift 2 --deliver ebu-r128,apple
```

Use this for a finished or rough stereo bounce, with vocals and beat already together. It
works like a mastering engineer:

1. **Diagnose** the mix and report the problems in plain words:
   - clipped bounce
   - lossy source file
   - DC offset
   - **out-of-phase bass** (it would vanish on phones)
   - boomy, muddy, harsh or dull tone
   - dead air at the start
2. **Repair:**
   - remove DC and rumble
   - recover out-of-phase bass and make it mono
   - de-ess and tame harshness **on the centre channel only** (where the lead vocal sits), so the
     beat's stereo content isn't touched
   - optional `--vocal-lift` to bring the vocal forward
   - trim dead air
3. **Master and verify,** exactly like the full pipeline. `--reference song.wav` matches the
   tone of a released track.

Auto-tune, de-essing a single vocal, and vocal-to-beat balance need the separate vocal file, so
use the main command for those. Pulling the vocal out of a finished mix takes an AI separation
model, which is too heavy to run on a phone.

## Teach it your sound (reference profiles)

```bash
studiomix learn maes1.mp3 maes2.mp3 ninho.wav --name maes      # once (add more songs any time)
studiomix vocal.wav beat.wav --profile maes                    # or
studiomix master mix.wav --profile maes
```

`learn` measures released songs you love and saves a **profile**: their loudness, their tonal
balance (below each file's lossy cutoff, so MP3s don't teach it a dull top end), and their stereo
width in four bands. Only the numbers are stored, never the audio. With `--profile`, the master:

- aims at the profile's loudness (kept between -16 and -7.5 LUFS; `--lufs` overrides it)
- matches its tonal balance in two passes. The second pass re-measures after the compressors,
  so the result really lands there.
- nudges each band's width toward the references, while the bass stays mono

The more references, the better the target. One song's arrangement colours its spectrum; an
average of 3–5 songs is the sound. In the app, open **Reference library**, add songs and tap
*Learn this sound*, then choose it under **Sound like**. `studiomix profiles` lists what's saved.

## Broadcast and platform versions

`--deliver` (on both commands, and as chips in the app) renders extra 24-bit versions from the
same master. Each one is checked against its spec with both meters:

| version    | for                                   | loudness       | true peak |
|------------|---------------------------------------|----------------|-----------|
| `ebu-r128` | EU radio / TV broadcast               | -23 LUFS ±0.5  | ≤ -1 dBTP |
| `atsc-a85` | US radio / TV broadcast               | -24 LKFS ±2    | ≤ -2 dBTP |
| `apple`    | Apple Music Sound Check level         | -16 LUFS ±0.5  | ≤ -1 dBTP |
| `streaming`| Spotify / YouTube reference level     | -14 LUFS ±0.5  | ≤ -1 dBTP |

Broadcast versions skip the clipper and use a slower limiter release, so they keep their dynamics.

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

### Phone studio: record with earbuds

In the app, choose **🎙️ Record**:

1. Name the song and pick the beat. Both are saved on the phone, so you can come back later.
2. **Calibrate once per pair of earbuds:** hold an earbud against the phone's mic and tap
   *Calibrate*. It plays clicks and measures the exact delay of that output (Bluetooth earbuds
   are often 150–300 ms late). Every take is shifted by that amount, so it sits exactly where
   you heard the beat.
3. Choose **Lead** or **Ad-lib**, set *start at* (to punch in a verse or hook; you hear 3 s of
   beat before it), tap **REC**, perform, tap **STOP**. Each take is saved as uncompressed
   audio the moment you stop, with a warning if it clipped or is too quiet. ▶ plays it back
   against the beat; 🗑 deletes it.
4. Tap **Mix & master**: your takes are laid on the beat's timeline and go through auto-tune,
   harmonies, doubles, the mix and the verified master.

The recorder captures the raw mic: the phone's call processing (echo cancellation, noise
suppression, auto-gain) is switched off, because it damages vocals.

**Earbuds and mics: what actually works**

- **Listening:** any earbuds. With Bluetooth, calibrate (step 2).
- **Recording:** the **phone's own microphone** is usually the best mic you have. Hold the phone
  like a mic, about a hand-width from your mouth, a bit off to the side to avoid pops. Wired
  earbuds with a mic are OK.
- **Avoid Bluetooth earbud mics.** When an app records through them, Bluetooth switches to
  phone-call mode: narrow, muffled sound. On many phones the beat in your ears also drops to
  call quality while you record. If that happens, use wired earbuds, or pick the phone's
  microphone in the *Microphone* list.
- Record in a small room with soft things around (clothes, a bed, curtains). Room echo can't
  be removed later; noise between phrases can be, and it is.

### The app (no typing)

```bash
studiomix serve
```

This opens studiomix in your phone's browser (in Termux it uses `termux-open-url`). Pick your
lead, beat and ad-libs with the normal file picker, choose the style, auto-tune, harmonies and
doubles, then tap **Mix & master**. You can follow the progress, listen to the result, and
download every file. Songs are also saved to **Downloads/studiomix**. The app only accepts
connections from the phone itself. `studiomix serve --host 0.0.0.0` lets a laptop on the same Wi-Fi
use it too. On a computer, the same command opens your normal browser.

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

**Vocal stack, Flex-Tune and key changes**
- `--harmony 3up,5down`: harmony voices *in key*. They follow your tuned melody, and a third is
  major or minor depending on where you are in the scale. Formants are kept, so it sounds like you
  singing the harmony, not a chipmunk. Intervals: `3up 3down 4up 5up 5down 6down 8up 8down`.
- `--doubles`: two double-tracks of the lead, panned wide. Each one wanders 5–22 ms late and a
  few cents sharp or flat, which is what makes stacked vocals sound thick.
- `--stack-at 0:45-1:15,2:10-2:40`: only stack on the hook (default: the whole song).
- `--flex 35`: Flex-Tune. Notes within 35 cents of the target get corrected. Bigger bends, falls
  and blue notes are treated as intentional and left alone.
- `--key-changes`: for songs that change key. It detects a key per section, with a cost for
  switching, so one borrowed chord doesn't flip the key.

**Deep voices:** pitch tracking goes down to 50 Hz. A first pass learns the singer's range, then
the search narrows to it, so deep voices keep their low notes and higher voices don't pick up
fake low pitches in breath noise.

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

### Release specs: verified, not assumed

- **Loudness meter** (ITU-R BS.1770-4 / EBU R128): passes the EBU Tech 3341 integrated and
  gating cases and the EBU Tech 3342 loudness-range cases within ±0.1 LU (in the test suite).
- **True-peak meter:** 4× oversampling plus parabolic peak refinement. On worst-case tones up to
  20 kHz, where every sample misses the peak, it reads within ±0.04 dB. Plain 4×
  oversampling under-reads by up to 0.13 dB.
- **On target:** the limiter search lands within 0.03 LU of the target and never above it.
- **The delivered files are checked, not the internal audio.** Every report decodes the 24-bit
  and 16-bit WAVs *from disk* (after dither and resampling) and measures them with our meter
  **and** ffmpeg's independent EBU R128 meter. A file only passes if both are within spec.

Measured on real material (real voices over a real hip-hop loop):

| preset    | file   | ours: LUFS / dBTP | ffmpeg: LUFS / dBTP | ceiling |
|-----------|--------|-------------------|---------------------|---------|
| trap      | 24-bit | -8.50 / -2.02     | -8.5 / -2.0         | -2.0    |
| trap      | 16-bit | -8.50 / -2.04     | -8.5 / -2.0         | -2.0    |
| pop       | 24-bit | -11.00 / -2.05    | -11.0 / -2.0        | -2.0    |
| streaming | 24-bit | -14.00 / -1.05    | -14.0 / -1.0        | -1.0    |

The MP3 preview always peaks 0.5–1 dB higher, because lossy encoding creates new peaks. This is
exactly why masters louder than -14 LUFS get a -2 dBTP ceiling. Upload the WAV, never the MP3.

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
| `--harmony 3up,5down` / `--harmony-level -9` | harmony voices in key / their level (dB) |
| `--doubles` / `--doubles-level -7` | double-tracked lead / their level (dB) |
| `--stack-at 0:45-1:15` | only stack harmonies/doubles in these parts |
| `--flex 35` | Flex-Tune: leave bends beyond 35 cents alone |
| `--key-changes` | detect a key per song section |

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
