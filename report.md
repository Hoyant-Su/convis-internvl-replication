# Persona effects relative to wording and sampling in InternVL3.5

Haoyang Su · Fudan University and Shanghai Innovation Institute  
[Research profile](https://hoyant-su.github.io/) · hoyantsu@gmail.com  
3 November 2026

## Result

InternVL3.5-8B produced greater variation across personas than across the two question wordings in the released ConVis experiment. The mean paired difference in cosine distance was **0.0478**, with a 95% confidence interval of **[0.0241, 0.0714]** across 43 complete images. The corresponding difference relative to repeated sampling was **0.0139 [−0.0069, 0.0346]**. These results distinguish the wording comparison from the sampling comparison when interpreting persona effects.

The experiment adds 768 responses from a third model configuration to the [September ConVis study](https://doi.org/10.5281/zenodo.22721059). It uses the same 48 images, prompts and requested seeds, enabling a direct replication of the experimental design.

## Experiment

Each COCO val2017 image was evaluated under four personas: an adult observer, a professional chef, a security professional and a wheelchair user. Each persona had two question wordings and two requested seeds, 11 and 29. Every request asked for three visible objects and a plausible use for each. The resulting design has 48 × 4 × 2 × 2 = 768 requests. The full schedule and exact prompts are in [data/design.json](data/design.json).

The model was [OpenGVLab/InternVL3_5-8B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF/tree/741a7d03020411e666c6109218ab71e08151ef86), using bfloat16 weights, Transformers 4.57.1 and PyTorch 2.8.0 with SDPA attention. Image preprocessing used the checkpoint processor. Sampling used temperature 0.7, top-p 1.0, top-k 0 and a maximum of 256 output tokens. The standard chat template was used. Each request reset its random seed and started with a fresh conversation.

The 48 images were divided among four GPU workers. Each worker ran all 16 conditions for its assigned images in the order inherited from the original schedule. All 768 requests reached an end token. Five responses had JSON parsing errors, leaving 763 compliant responses and 43 images with all 16 responses compliant. The five errors occurred in three chef responses and two wheelchair-user responses. Every generated response is retained in [results/internvl](results/internvl).

## Analysis

Object names and uses were concatenated, lowercased and encoded with the original study's all-MiniLM-L6-v2 checkpoint. Encoding used attention-mask mean pooling, unit normalization and a maximum length of 256 tokens. For each image, persona pairs shared wording and seed, wording pairs shared persona and seed, and sampling pairs shared persona and wording. Distances were averaged within each pair category before calculating the paired contrasts.

The primary analysis includes images with 16 compliant responses. An image is the unit of inference. Confidence intervals use the Student t distribution across image-level differences. The available-pair analysis retains usable response pairs from all 48 images. Raw-word Jaccard distance provides the second similarity measure specified by the source analysis.

| Analysis | Images | Persona minus wording | Persona minus sampling |
| --- | ---: | ---: | ---: |
| Cosine distance, complete images | 43 | 0.0478 [0.0241, 0.0714] | 0.0139 [−0.0069, 0.0346] |
| Cosine distance, available pairs | 48 | 0.0497 [0.0281, 0.0713] | 0.0139 [−0.0054, 0.0331] |
| Word Jaccard distance, complete images | 43 | 0.0573 [0.0249, 0.0897] | −0.0026 [−0.0277, 0.0225] |
| Word Jaccard distance, available pairs | 48 | 0.0656 [0.0354, 0.0957] | 0.0000 [−0.0235, 0.0236] |

Values are means and 95% confidence intervals. The median image-level persona-minus-wording cosine contrast was 0.0438. The full image-level estimates, pair distances and response classifications are supplied as CSV files in [the analysis directory](results/internvl/analysis).

The analysis was also run on both sets of released reference responses before collecting the new data. Response classifications and complete-image sets matched exactly. Across the cosine and word-distance summaries, the largest absolute difference from the published means, standard deviations and confidence limits was below 5 × 10⁻⁸.

| Configuration | Complete images | Persona minus wording cosine distance |
| --- | ---: | ---: |
| Released Qwen3.5-9B configuration | 37 | 0.0146 [−0.0003, 0.0295] |
| Released Ollama llava:13b configuration | 32 | −0.0266 [−0.0412, −0.0121] |
| InternVL3.5-8B replication | 43 | 0.0478 [0.0241, 0.0714] |

The three rows describe their respective complete-image populations. Model, precision and inference backend vary across configurations. InternVL's positive wording contrast extends the configuration-dependent pattern in the September study. Its sampling contrast remains uncertain. The measurements concern linguistic reports on this image sample; their relation to object-use correctness or embodied behavior remains an open empirical question. The primary estimates are conditional on complete responses, while the available-pair analysis shows how the result changes with broader response coverage.

## Runtime and reproduction

Collection used four NVIDIA H100 80GB GPUs, with one model replica and two CPU threads per worker. After model loading, collection took 399.1 seconds across the four workers, with 25.9 summed worker-minutes and a peak allocated GPU memory of 18.01 GB per worker. A single H100 can run the workers' requests sequentially. Four CPU cores and 32 GB of host memory are a recommended starting configuration for a single-worker rerun. The recorded software versions are in [requirements.txt](requirements.txt).

Install the requirements in an existing compatible environment and obtain the checkpoints:

```bash
hf download OpenGVLab/InternVL3_5-8B-HF --revision 741a7d03020411e666c6109218ab71e08151ef86 --local-dir /path/to/InternVL3_5-8B-HF
hf download sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --local-dir /path/to/all-MiniLM-L6-v2
```

To repeat collection and analysis on four GPUs:

```bash
bash reproduce.sh /path/to/InternVL3_5-8B-HF /path/to/all-MiniLM-L6-v2 4 rerun
```

The third argument sets the worker count. Use `1` for a sequential single-GPU run. Completed requests are retained when resuming with the same worker count and output directory.

To recompute the submitted analysis from stored responses on CPU:

```bash
python analyze.py --embedding-path /path/to/all-MiniLM-L6-v2 \
  --responses results/internvl/part-*.jsonl --label internvl --output rerun-analysis
```

To reproduce a released reference configuration:

```bash
python analyze.py --embedding-path /path/to/all-MiniLM-L6-v2 \
  --responses data/reference/qwen35.jsonl --label qwen35 \
  --reference qwen35 --output rerun-reference
```

## Sources

The design, prompts and reference responses come from Murad Farzulla, *Context-Dependent Affordance Reports in Vision-Language Models*, version 3.0.0, 12 September 2026, [Zenodo](https://doi.org/10.5281/zenodo.22721059), CC BY 4.0. This contribution adds the InternVL3.5 experiment, a native Transformers runner and an independent implementation of the image-level analysis.

Images are the 48 [COCO val2017](https://cocodataset.org/#download) files named in the released schedule. `fetch_images.py` retrieves those files individually. Model documentation is available from [OpenGVLab](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF) and [Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
