# studiomix

Give it a **vocal** and an **instrumental**. You get back a mixed, mastered song that is ready to
upload to Spotify, Apple Music, YouTube, TikTok, or any distributor (DistroKid, TuneCore, CD Baby…).

```bash
studiomix vocal.wav beat.wav --preset hiphop --name "My Song"
```

```
out/
  My Song_master_24bit_48k.wav        <- upload this to your distributor
  My Song_master_16bit_44.1k.wav      <- CD-quality / for distributors that require 16-bit
  My Song_master_preview_320k.mp3     <- for sharing with people (don't upload it to stores)
  My Song_premaster_mix_24bit.wav     <- the mix before mastering, peaks at -6 dBFS (for a mastering engineer)
  My Song_vocal_stem_24bit.wav        <- processed vocal on its own
  My Song_instrumental_stem_24bit.wav <- processed beat on its own
  My Song_report.txt / .json          <- loudness, peaks, what was done, delivery checks
```

## Install

Requires Python 3.10+.

```bash
pip install -e .          # or: pip install -e ".[dev]" to run the tests
```

`wav`, `flac`, `aiff`, `ogg` and `mp3` files load out of the box. Install `ffmpeg` to open other
formats like `m4a`.

## What it does

**Lead vocal**
1. Folds a stereo vocal to mono, then sets a fixed working level so every later stage behaves the same on every take
2. 4th-order high-pass filter to remove rumble, plosives and mic-handling noise
3. Soft gate (downward expander) to clean up room noise between phrases
4. **Adaptive EQ**: measures the vocal's spectrum and corrects it toward a balanced lead-vocal curve. It cuts up to 6 dB but boosts at most 3 dB.
5. Mud cut (300 Hz) and boxiness cut (800 Hz), then presence (3.5 kHz) and air (12 kHz) boosts
6. **Two-stage compression**: a fast peak catcher feeding a slow, smooth leveler
7. Split-band **de-esser** with a threshold that adapts to the singer
8. Parallel tape-style saturation, oversampled, for warmth
9. **Vocal rider**: automatic fader moves that keep the vocal at the same level relative to the beat, so it doesn't get buried in loud choruses
10. Tempo-synced ping-pong delay and a plate-style reverb. Both are ducked by the dry vocal so the words stay clear.

**Instrumental**
- Subsonic filter
- **Vocal-keyed dynamic EQ**: the beat dips a few dB in the 1.5–5 kHz range only while the vocal is singing. This makes space for the vocal without turning the beat down.

**Master bus**
1. Tonal balance: either matches a **reference song** (`--reference`) or nudges the mix toward the spectral tilt of commercial releases
2. Phase-coherent 3-band compression (Linkwitz-Riley crossovers)
3. Bus "glue" compressor
4. Mono low end below 120 Hz, slightly wider highs
5. Oversampled soft clipper, then a look-ahead **true-peak limiter**. Loudness is adjusted repeatedly until the integrated loudness hits the target within ±0.1 LU.
6. The 44.1 kHz master gets its own limiting pass (instead of resampling an already-limited file), then TPDF dither for the 16-bit export

All loudness readings are ITU-R BS.1770-4 / EBU R128 LUFS. All peak readings are 4×-oversampled true peak.

## Presets

| preset      | sound                                                     | loudness |
|-------------|-----------------------------------------------------------|----------|
| `pop`       | bright, upfront vocal, polished (default)                 | -11 LUFS |
| `hiphop`    | dry, loud, in-your-face vocal, heavy mono low end         |  -9 LUFS |
| `rnb`       | warm vocal, lush reverb, wider image                      | -11 LUFS |
| `rock`      | vocal sits inside dense guitars, more bite                | -10 LUFS |
| `acoustic`  | natural and dynamic, light compression                    | -14 LUFS |
| `streaming` | exactly the Spotify/YouTube reference level, max dynamics | -14 LUFS |

**About loudness and Spotify:** Spotify, YouTube, Apple Music and Tidal turn every song to roughly
-14 LUFS. A louder master doesn't play louder. It gets turned down and keeps its denser, punchier
character. Following Spotify's own guidance, any master louder than -14 LUFS is automatically
limited to **-2 dBTP** instead of -1 dBTP, so it doesn't distort when encoded to Ogg/AAC. You can
override this with `--ceiling`.

## Fine tuning

```bash
studiomix vox.wav beat.mp3 -p rnb \
  --vocal-level 1.5      # vocal louder vs. the beat (dB)
  --reverb 0.2           # more reverb (0-1)
  --delay 0.1            # more delay (0-1)
  --air 4                # brighter top end on the vocal (dB)
  --vocal-comp 1.4       # heavier vocal compression (0-2)
  --deess 10             # stronger de-essing (dB, 0 = off)
  --carve 4              # beat makes more room for the vocal (dB)
  --lufs -10             # loudness target
  --reference "Drake - song.wav"   # match the tone of a song you like
  --offset-ms 25         # vocal recorded late/early? shift it
```

Run `studiomix --help` for everything.

## Tips for the best result

- Export the vocal and the beat **from the same start point** (bar 1), as WAV, with no effects or limiter on the vocal
- Leave headroom: no clipping in either file. Anything that peaks below about -3 dBFS is fine.
- If the beat is already mastered and loud, that's OK. The chain re-balances it.
- Try `--reference` with a professionally released song in the same genre for the closest match to "that sound"

## Try it without your own files

```bash
python tools/make_demo.py demo/
studiomix demo/demo_vocal.wav demo/demo_beat.wav -p hiphop
```

## Tests

```bash
pytest
```
