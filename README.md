## Homomorphic KV Quantization

Homomorphic KV Quantization (HKVQ) for Disaggregated LLM Serving

## Datasets

### Accuracy Metric Selection

- For classification tasks and information retrieval tasks, we use the ***accuracy*** as the metric.
- For summarization tasks, we use ***ROUGE-1*** [1] score as the accuracy score.
- For code completion, we use ***Edit Similarity (normalized Levenshtein distance)*** [2-3] as the accuracy.

### Dataset Dir
IMDb movie genre classification: /datasets/imdb

arXiv summarization: /datasets/arxiv

Cocktail for information retrieval: /datasets/cocktail

HumanEval for code completion: /datasets/humaneval

## HKVQ Dir
/
- vllm  # vLLM base code.
- datasets  # The datasets used in the paper.
- exp  # Model code used in the paper for exp.
- kernels  # Self-attention kernels.
- quantization  # Code for quantization test.

```
vllm: the root dir has the vLLM base code.
datasets: it contains the datasets we use for validation.
exp: it has the implementation code of for different models and prefill_decode disaggregated code.
kernels: kernel functions.
quantization: it has the code of quantization methods.
```

## References
[1] ROUGE Score, https://en.wikipedia.org/wiki/ROUGE_(metric)

[2] Zhang, Lei, et al. "Hierarchical Context Pruning: Optimizing Real-World Code Completion with Repository-Level Pretrained Code LLMs." arXiv preprint arXiv:2406.18294 (2024).

[3] String Similarity Metrics - Edit Distance, https://www.baeldung.com/cs/string-similarity-edit-distance
