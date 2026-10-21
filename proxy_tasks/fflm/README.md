# E3 FFLM — one-seed pilot

For EvoScientist architecture search, use `python -m validation` from the repository
root; see `docs/EVOSCIENTIST.md`. That interface scores ID development data only.
Direct execution below retains the original ID/sparse/dense final evaluation.
`FFLM_DATA_DIR` can point to existing verified data. Data/vendor assets are not
tracked by Git; `package_manifest.json` describes the original bundle, and the
restore/check tools verify its data/vendor entries. Modified driver code is
identified by Git and the validation source/evaluator hashes.

This package reconstructs the deterministic (clean) task described in Liu et al. 2023, https://arxiv.org/html/2306.00946v2. It is not the authors' original training code or a 500-seed replication.

Training: T=512, fresh FFL(0.8) samples, batch16; AdamW beta=(.9,.999), lr3e-4, wd0.1, 50-step warmup, linear decay with step10001=0. Only next-token targets following read instructions contribute to cross entropy. No early stopping or selection using OOD results.

Transformer: x-transformers1.27.19, 6 layers, dim512, 8 heads, head dim64, FFN4x/GELU, pre-norm, learned absolute positional embeddings, zero dropout, untied output, PyTorch SDPA, FP32. Actual parameters19,163,648. Original-paper version, PE/normalization details, AdamW parameter groups and arithmetic precision were not completely specified; these are explicit reconstruction choices. Changes in implementation can change failures.

LSTM: learned5x128 embedding, one128-hidden layer, five-token linear head, FP32. Actual parameters133,381. The embedding is an explicit reconstruction to match the paper's stated133K size. It trains500steps; the LR decay endpoint remains10001 because the paper says the same optimizer hyperparameters. All parameters receive decay.

Model seed1729, training-data seed1000. Both architectures receive identical initial training batches. Transformer10,000steps and LSTM500steps follow the paper's unequal learning budgets; parameters/FLOPs/time are not matched.

Evaluation: fixed official dataset revision91ef5c91274612fd3fe3ac18da5c7c820a39378e, exact prefixes: ID1000sequences/26508reads, sparse100000/353875, dense3000/345781. Training samples follow the stated distribution but are not the released training-prefix samples. Both final models see identical test arrays. Each read is scored against five-vocabulary argmax, not binary-restricted argmax. Report error counts/rates and erroneous-sequence counts; zero observed errors does not prove correctness on all possible inputs. Report sparse/dense/joint zero separately. ID failure is retained rather than excluded.

Periodic ID evaluation every1000Transformersteps is monitoring only; final checkpoint is evaluated once across all three full sets. OOD is not evaluated during training. This differs from the paper's training-curve monitor frequency. Checkpoints are saved at1000-step intervals; final.pt is saved before final evaluation. No automatic resume is implemented. Launch attempts have distinct directories and IDs.

Run on server: bash start_server.sh

The launcher runs tiny GPU smoke checks first, then100-step isolated GPU timing benchmarks for both models, then a full1-seed LSTM/Transformer pair sequentially. Benchmarks do not continue into final training. Their accuracy is not an experimental result. Full run summaries are in runs/runs.jsonl, runs/<id>/result.json, history.jsonl, config.json and env.json. GPU timings and final evaluation costs are recorded. Errors are logged separately and halt the launcher; no CPU fallback occurs. Dependencies are vendored and do not modify the server's installed packages. Existing server torch2.5.1 is required; its CUDA availability is checked on launch.

Local validation: CPU smoke with torch2.12.0, oracle-checked evaluation arrays, independent state-machine validation of generated data, causal future-token invariance for both models. GPU throughput and full model quality have not yet been measured. Local smoke accuracy is not reproduction evidence.
