# Video model, prompt and candidate rules

## Which model

The owner names the video model, and `generate --model-id` takes that exact id. If the owner delegates the
choice by a stated criterion (for example a bake-off for the best performance per price), apply that criterion,
show the owner the result, and record it. With neither, ask. The project's image-model rule (the harness picks
from a quality/price chart) covers still images only; it does not choose a video model.

`generate` refuses a model that cannot start from a first frame, lacks the brief's `videoSeconds`, or lists no
size for its `resolution` and `aspectRatio`. Each candidate records the model that actually served it
(`modelReturned`); read it from `generate/record.json`.

## Cost

The estimate comes from the model's listed `video_tokens` price. A model with no listed price needs
`--max-cost`. `generate` refuses when the estimate is above `--max-cost`, or when 1.5 times it exceeds the
OpenRouter key's remaining limit, and checks again before each further candidate. A failure mid-run writes
`generate/record.partial.json` with what was spent. Two candidates per clip (`--candidates 2`); candidate N uses seed
`--seed` + N (1001 and 1002 by default).

## The prompt

`reference` renders our own model as the first frame, so the video starts from a known camera. The prompt
(`anim/prompt_template.txt`, filled from the brief's `subject`, `motion` and `gait`) asks for: a strict side
profile for the whole clip, the camera tracking alongside at the animal's speed and keeping it centred and the
same size, one continuous shot with no cuts, zoom, rotation or change of angle, exactly four legs all visible,
the same animal, coat, markings and proportions, and a plain flat background in the reference's colour with no
floor, shadow, scenery or text. Change the brief, not the template.

## Accepting a candidate

Open each candidate's strip (`generate/candidate-N-strip.png`) and several full frames of its video, then write
the checks file from what you saw:

```json
{"singleAnimal": true, "noCuts": true, "sideViewHeld": true, "noZoom": true, "legCount": true,
 "sameAnimal": true, "plainBackground": true, "reviewer": "agent", "notes": "frames 0, 30, 60, 90, 119 opened; ..."}
```

Every check must be true, with a reviewer and notes, or `accept` refuses. Reject a candidate when a second
animal appears, the shot cuts or jumps, the camera turns or zooms, a leg appears or vanishes, the coat or
proportions change, or scenery or a floor appears. Accept the other candidate or generate again; never fit a bad
video.

Seedance-type models return sound. `ingest` saves it and `export` bundles it as `candidate-audio.wav`; this
skill never wires it into the game.
