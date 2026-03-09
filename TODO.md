# SpaMo — TODO

## High Priority

- [ ] **Increase `num_workers`** — Currently set to `0` to prevent OOM. Once memory usage is profiled, experiment with `num_workers: 2-4` for faster data loading (monitor system RAM).

## Medium Priority

- [ ] **Evaluate on dev set** — Run evaluation on the dev split (`mode: dev`) and compare with test results.
- [ ] **Experiment with beam search parameters** — Current: `num_beams=5`, `top_p=0.9`, `do_sample=True`. Try pure beam search (`do_sample=False`) or adjust beam count.
- [ ] **Try `batch_size: 2-4`** — Monitor VRAM with `nvidia-smi` and increase if headroom exists.
- [ ] **Multi-dataset evaluation** — Add CSL-Daily and How2Sign dataset configs and evaluate.

## Low Priority / Future Work

- [ ] **Fine-tune on Phoenix14T** — Train with the current features to adapt the model.
- [ ] **WandB online logging** — Set `WANDB_MODE=online` for cloud experiment tracking.
- [ ] **Remove zero-padding workaround** — Once proper 1024-dim motion features are extracted, remove the padding logic in `spamo/t5_slt.py:prepare_visual_inputs()`.

## Experimental (Not Currently Required)

- `format_fixer.py` — Experimental file, not needed for current evaluation pipeline.
- `live_spamo.py` — Experimental live inference file, not needed for current evaluation pipeline.
