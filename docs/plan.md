# scene-to-score project plan

Version of 15 October 2026. Working assumption: six weeks, three or four people. To be adjusted
once the deadline and the group size are fixed.

## The question

From the representation of a film scene, can we predict the representation of its music well
enough to retrieve it among others, and to generate music that resembles it?

A single scene admits several musics. A naive prediction returns the average music. The project
compares ways to avoid that collapse.

## The pipeline

1. Film clip, cut into 10 s segments.
2. Frozen video encoder: an image encoder applied frame by frame (DINOv2), and V-JEPA 2.
3. Trained projection into the music space. Three objectives compared: regression, contrastive,
   and optionally a small generative model over the vectors.
4. CLAP music vector, from the CLAP model shipped with MusicLDM.
5. Frozen MusicLDM, which generates 10 s of audio from the predicted vector.

Training-free baseline: describe the scene in one sentence, then give that sentence to the same
generator.

## What runs where

| Machine | Use |
|---|---|
| Laptop (RTX 4070, 8 GB; about 30 GB free under WSL) | Development and tests on 20 to 30 films |
| Tech lab (up to 48 GB of GPU memory) | Full download (180 GB), full extraction, multi-seed training, generation |

The code is identical on both machines. Only the data location (`STS_DATA_DIR`) and the batch
size change.

To clarify on the tech lab before week 2:

- system and job scheduler (direct SSH access or a queue);
- disk quota, for the 180 GB of data and a few gigabytes of features;
- internet access from the compute nodes, for Hugging Face and uv;
- exact GPU model.

## Schedule

| Week | Tasks | Machine | Milestone |
|---|---|---|---|
| 1 | Repository scaffold and subset download (done); listen to 50 clips; tech lab environment | Laptop, tech lab | The audio track is mostly music |
| 2 | Extract video and music vectors, measure time per segment; text baseline | Laptop then tech lab | Vectors cached for the whole dataset |
| 3 | Regression and contrastive projections; rank-based evaluation; collapse diagnosis | Tech lab | First objectives × encoders table |
| 4 | Plug into MusicLDM; generation on the test set; start the report | Tech lab | Demo, first version |
| 5 | Full evaluation with three seeds; out-of-domain test on V2M-bench; listening test | Tech lab | All figures |
| 6 | Report, defence, final demo, repository cleanup | Laptop | Code frozen at the start of the week |

If running late, cut in this order: the generative model over vectors, the listening test, the
out-of-domain test. The objectives × encoders table is never cut.

## Team split, to assign

| Role | Contents |
|---|---|
| Data and evaluation | Subsets, sanity listening, rank and dispersion metrics, listening test |
| Encoders and extraction | DINOv2, V-JEPA 2, CLAP, caching, timing |
| Projection and training | Objectives, model selection, sweeps on the tech lab |
| Generation and demo | MusicLDM, text baseline, demo |

Everyone writes the report from week 4.

## Estimated compute

Estimates, not measurements, for the whole of OSSL-v2 (about 85,000 segments of 10 s) on the
laptop. The tech lab is several times faster.

| Step | Estimated time |
|---|---|
| Image encoder (1 frame per second) | 1 to 2 h |
| V-JEPA 2 ViT-L (16 frames per segment) | 2 to 4 h |
| CLAP on the music | under 30 min |
| 18 projection trainings (3 objectives × 2 encoders × 3 seeds) | 2 to 5 h in total |
| Generation of about 1,000 clips | 1 to 4 h |

The bottlenecks are disk and bandwidth, not the GPU. Measure time per segment on about a hundred
segments in week 2, then extrapolate.

## Technical notes

- **Clip ids**: start and end in hundredths of a second, checked on one clip.
- **OSSL-v2 audio**: mono, 44.1 kHz. MusicLDM's CLAP expects 48 kHz: resample.
- **MusicLDM**: the pipeline lives in `diffusers.pipelines.deprecated` (diffusers 0.41). With
  transformers 5, `ClapModel.get_text_features` returns an output object (`.pooler_output`), not
  a tensor, which the pipeline does not expect. Pass `prompt_embeds` and `negative_prompt_embeds`
  yourself, the latter being the vector of the empty text. Found by reading the code of the
  installed versions, not yet run.
- **CLAP vectors**: text and audio come out normalised (unit norm) in transformers 5.
- **V-JEPA 2**: `get_vision_features(pixel_values_videos)` in transformers. Average the tokens
  over space before storing them: about 4 MB per segment otherwise, against 16 KB.
- **Video decoding**: ffmpeg is not installed under WSL. PyAV ships its own libraries and avoids
  depending on it.
- **Segments of the same clip** share their music. At evaluation, keep only one per clip; in
  contrastive training, they create false negatives.

## Risks

1. Old films (62% from the 1930s and 1940s): transfer to recent videos is uncertain. The
   V2M-bench test measures it.
2. The separated music still contains non-musical sounds, according to the dataset authors.
3. Regression will probably collapse to the mean: that is the point of the comparison.
4. MusicLDM was trained on recent music; the rank metric does not depend on it.
5. Music similarity is computed in the space used for prediction; the listening test is an
   independent check.

## Done when

- The objectives × encoders table exists, on OSSL-v2 and V2M-bench, with three seeds.
- The demo runs: a scene, the generated music, the original music and the text baseline's music.
- The repository runs from scratch with the README commands.
- The report is four to six pages long.
