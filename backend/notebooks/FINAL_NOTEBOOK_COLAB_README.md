# PeptiScout Final Notebook Colab Run Notes

Use `final_peptiscout_notebook.ipynb` as the final technical notebook artifact.

## Colab Setup

1. Open the notebook in Google Colab.
2. Set runtime to A100 GPU.
3. Put these files in `/content/drive/MyDrive/Pepti_scout/`:
   - `peptide_dataset.json`
   - `benchmark_100.json`
   - optionally `results.json`
   - optionally `run_evaluation.py`
4. Ensure your Hugging Face account has access to `meta-llama/Llama-3.2-11B-Vision-Instruct`.
5. Run the setup, scoring, baseline, 4-bit QLoRA, inference, and export sections.
6. For final submission, set `RUN_FINAL_INFERENCE = True`, set `FINAL_RUN_LIMIT = None`, set `RUN_CA_VALIDATION = True` if you want PMID validation, then save the notebook with outputs visible.

## Final Outputs

The notebook is set up to export into `/content/drive/MyDrive/Pepti_scout/`:

- `predictions_llama_11b.json`
- `results_llama_11b.json`

The final comparison table should use same-Llama rows:

- prompt-only Llama baseline
- 4-bit QLoRA fine-tuned Llama
- Llama tool-agent variant

## Optional Repo Scoring

After downloading `predictions_llama_11b.json`, score it without FastAPI:

```bash
python -m backend.scripts.run_evaluation \
  --predictions predictions_llama_11b.json \
  --output backend/data/results_llama_11b.json \
  --benchmark backend/data/benchmark_100.json \
  --base-model meta-llama/Llama-3.2-11B-Vision-Instruct \
  --quantization 4bit-nf4 \
  --model-family llama
```
