# MLPerf Inference - Step 1 Support Analysis

## Summary Table
| Process | Support Level | Evidence |
|---------|--------------|----------|
| S1P1 | 2 | Fixed benchmark tasks with predefined metrics; limited custom task definition |
| S1P2 | 2 | Supports objective metrics and multiple scenarios (offline/server); limited subjective assessment |
| S1P3 | 2 | Standard benchmarks provided per model; custom datasets possible with significant integration work |
| S1P4 | 3 | Excellent multi-backend support (PyTorch, vLLM, SGLang, TensorFlow, ONNX); configurable via registry |
| S1P5 | 1 | Basic model inference only; no LLM-as-judge or human-in-the-loop evaluation |
| S1P6 | 2 | MLPerf.conf provides test duration/seeds/constraints; limited rate limiting and cost tracking |

**Overall Score: 12/18 (67%)**

---

## Detailed Analysis

### S1P1: Define evaluation tasks and success criteria
**Rating:** 2 (Partial Support)

**Evidence:**

The MLPerf Inference benchmark provides a **fixed set of benchmark tasks** rather than a flexible task definition system:

1. **Predefined Tasks by Model:**
   - Source: `README.md` lines 27-42 show fixed model-benchmark pairs:
     - resnet50-v1.5 → ImageNet classification
     - bert → SQuAD-1.1
     - dlrm-v2 → Criteo Terabyte recommendation
     - llama2-70b, llama3.1-405b, mixtral-8x7b → OpenOrca
     - deepseek-r1 → AIME, MATH500, gpqa, MMLU-Pro, livecodebench
     - whisper → LibriSpeech

2. **Success Criteria Defined in mlperf.conf:**
   - Source: `mlperf.conf` lines 1-100
   - Performance sample counts: `deepseek-r1.*.performance_sample_count_override = 4388`
   - Seeds for reproducibility: `*.*.qsl_rng_seed = 1780908523862526354`
   - Scenario-specific constraints: `*.Server.target_latency = 10`, `*.Server.target_latency_percentile = 99`
   
3. **Accuracy Targets in submission_checker.py:**
   - Source: `tools/submission/submission_checker.py`
   - `Config` class has methods like `get_accuracy_target()`, `get_accuracy_upper_limit()`
   - Hardcoded per-model accuracy requirements for validation

4. **Task-Specific Metrics:**
   - Language models: ROUGE scores (see `language/llama2-70b/evaluate-accuracy.py`)
   - Vision: ImageNet accuracy, COCO mAP
   - Recommendation: AUC-ROC (see `retired_benchmarks/recommendation/dlrm/tf/roc_metrics/`)

**Limitations:**
- **No custom task registry** - Cannot easily add new evaluation tasks
- **No flexible metric definitions** - Each benchmark has hardcoded metrics
- Tasks are **tightly coupled** to specific models and datasets
- Success criteria defined in multiple locations (mlperf.conf, submission_checker.py, per-benchmark scripts)
- No unified configuration for defining evaluation objectives

**Why Partial Support:**
- ✓ Tasks are well-defined and documented
- ✓ Success criteria exist for correctness and performance
- ✗ Not configurable - must use predefined task-model-dataset triplets
- ✗ Adding custom tasks requires significant code changes across multiple files

---

### S1P2: Select evaluation methodologies
**Rating:** 2 (Partial Support)

**Evidence:**

1. **Objective Metrics - Well Supported:**
   - Source: `loadgen/test_settings.h` lines 76-81
   - Four test scenarios: SingleStream, MultiStream, Server, Offline
   - Each has objective performance metrics (latency percentiles, QPS, throughput)

2. **Test Modes:**
   - Source: `loadgen/test_settings.h` lines 84-98
   - `AccuracyOnly`: Runs each sample through SUT, outputs to accuracy JSON
   - `PerformanceOnly`: Measures performance traffic patterns
   - `SubmissionRun`: Combines accuracy + performance
   - `FindPeakPerformance`: Determines maximum QPS

3. **Evaluation Paradigms:**
   - Classification: Supported (ResNet50, BERT)
   - Generation: Supported (GPT-J, LLaMA, DeepSeek-R1)
   - Ranking/Recommendation: Supported (DLRM-v2)
   - Detection: Supported (RetinaNet)
   - Retrieval: Not explicitly supported

4. **Batch vs Real-time:**
   - Source: `mlperf.conf` and test scenarios
   - Offline scenario = batch evaluation
   - Server/SingleStream = real-time evaluation
   - Configurable via scenario selection

**Limitations:**
- **No subjective assessment** - No human evaluation or LLM-as-judge
- **No A/B testing framework** - Cannot compare multiple models systematically
- Limited support for **retrieval quality** evaluation
- No built-in **human annotation interfaces**

**Why Partial Support:**
- ✓ Strong support for objective metrics
- ✓ Multiple evaluation modes (accuracy, performance, combined)
- ✓ Both batch and real-time scenarios
- ✗ No subjective/human evaluation
- ✗ No LLM-as-judge methodology

---

### S1P3: Choose appropriate datasets and benchmarks
**Rating:** 2 (Partial Support)

**Evidence:**

1. **Built-in Standard Benchmarks:**
   - Source: `README.md` and per-benchmark directories
   - **Language:** SQuAD-1.1 (BERT), CNN-DailyMail (GPT-J), OpenOrca (LLaMA), MMLU-Pro/MATH500/AIME (DeepSeek-R1)
   - **Vision:** ImageNet2012, OpenImages, COCO 2014
   - **Recommendation:** Criteo Terabyte
   - **Medical:** KiTS19, BraTS19
   - **Speech:** LibriSpeech
   - **Graph:** IGBH
   - **Automotive:** Waymo Open Dataset

2. **Dataset Loading Infrastructure:**
   - Source: `language/deepseek-r1/utils/data_utils.py`
   - `load_dataset()` function for loading preprocessed datasets
   - Validation utilities: `validate_dataset()`, `validate_dataset_extended()`
   - Per-benchmark dataset classes (e.g., `language/gpt-j/dataset.py`)

3. **Custom Dataset Support:**
   - Source: Various `dataset.py` files
   - Each benchmark has a `Dataset` class for loading/preprocessing
   - Format: Typically pandas DataFrame pickles (`.pkl` files)
   - Example: `language/llama3.1-8b/dataset.py` shows `Dataset` class with `__init__`, `__len__`, `__getitem__`

4. **Preprocessing Scripts:**
   - Per-benchmark preprocessing (e.g., tokenization, format conversion)
   - Calibration dataset support (see DeepSeek-R1 README)

**Limitations:**
- **No unified dataset registry** - Each benchmark has its own dataset loader
- **No synthetic data generation** - Must provide real datasets
- **Limited cross-benchmark reusability** - Dataset loaders are benchmark-specific
- **No built-in data augmentation** or sampling strategies
- Custom datasets require **implementing benchmark-specific interfaces**

**Why Partial Support:**
- ✓ Comprehensive set of standard benchmarks
- ✓ Clear dataset loading patterns
- ✓ Custom datasets technically possible
- ✗ No unified dataset API or registry
- ✗ No synthetic data generation
- ✗ Significant integration work for custom datasets

---

### S1P4: Configure evaluation infrastructure
**Rating:** 3 (Good Support)

**Evidence:**

1. **Multi-Backend Support - Excellent:**
   - Source: `language/deepseek-r1/backends/` directory
   - **PyTorch:** Native PyTorch inference (`pytorch_backend.py`)
   - **vLLM:** High-performance LLM serving (`vllm_backend.py`)
   - **SGLang:** SGLang inference engine (`sglang_backend.py`)
   - TensorFlow, ONNX Runtime in retired benchmarks
   - Base backend abstraction: `base_backend.py` with `BaseBackend` ABC

2. **Backend Registry System:**
   - Source: `language/deepseek-r1/utils/backend_registry.py`
   - `BACKEND_REGISTRY` dict with full metadata per backend
   - Configuration per backend: model_name, batch_size, temperature, dtype, etc.
   - Environment variables: `env_vars` dict for each backend
   - Compatibility matrix: `compatible_runners` list per backend

3. **Resource Configuration:**
   - Source: `backend_registry.py` lines 40-122
   - Tensor parallelism: `tensor_parallel_size: 8`
   - GPU memory: `gpu_memory_utilization: 0.90`
   - Batch sizes: `max_num_seqs: 64`, `batch_size: 16`
   - Sequence lengths: `max_input_len`, `max_output_len`, `max_model_len`

4. **Distributed Execution:**
   - Source: `language/deepseek-r1/README.md` lines 128-139
   - PyTorch backend uses `torchrun --nproc_per_node=8`
   - MPI support: `run_eval_mpi.py`, `run_mlperf_mpi.py`
   - Distributed configuration via torchrun/mpirun

5. **Docker Containerization:**
   - Source: `language/deepseek-r1/launch_docker.sh`
   - Backend-specific containers
   - Volume mounting for models/data
   - User management and GPU access

6. **Model Support:**
   - Source: Various benchmark READMEs
   - OpenAI API-compatible endpoints (vLLM, SGLang)
   - HuggingFace models
   - Local checkpoints
   - Custom model implementations

**Limitations:**
- **No cloud provider integrations** (AWS Bedrock, Azure OpenAI, etc.) - only self-hosted
- **No automatic resource scaling** or dynamic allocation
- Sandboxing limited to Docker containers (no fine-grained code execution sandboxing)

**Why Good Support:**
- ✓ Multiple backend adapters with clean abstraction
- ✓ Comprehensive configuration system via registry
- ✓ Resource allocation controls (GPU memory, batch size, parallelism)
- ✓ Docker containerization for isolation
- ✓ Distributed execution support
- ✓ Well-documented backend selection and configuration

---

### S1P5: Design evaluator pipeline
**Rating:** 1 (Basic Support)

**Evidence:**

1. **Rule-Based Evaluation:**
   - Source: Per-benchmark accuracy scripts
   - `language/llama2-70b/evaluate-accuracy.py`: Computes ROUGE scores
   - `language/deepseek-r1/eval_accuracy.py`: Task-specific evaluators (LiveCodeBench, MATH500, etc.)
   - Hardcoded evaluation logic per benchmark

2. **Evaluation Functions:**
   - Source: `language/deepseek-r1/eval_accuracy.py` lines 517-658
   - `evaluate_livecodebench()`: Code generation evaluation
   - Task-specific pass/fail checks
   - No abstraction for custom evaluators

3. **Metric Computation:**
   - Source: `retired_benchmarks/translation/gnmt/tensorflow/nmt/utils/evaluation_utils.py`
   - `evaluate()` function with metric parameter: "bleu", "rouge", "accuracy", "word_accuracy"
   - Limited to predefined metrics

**Limitations:**
- **No LLM-as-judge** - Cannot use language models for evaluation
- **No human-in-the-loop** - No annotation interfaces or human review
- **No custom evaluator API** - Must modify benchmark-specific code
- **No evaluator composition** - Cannot chain multiple evaluators
- **No prompt-based evaluation** - No support for few-shot evaluation or rubrics
- Evaluation logic is **tightly coupled** to each benchmark

**Why Basic Support:**
- ✓ Rule-based evaluation exists
- ✓ Per-benchmark accuracy computation
- ✗ No LLM-as-judge
- ✗ No human evaluation support
- ✗ No extensible evaluator framework
- ✗ No custom evaluator APIs

---

### S1P6: Set up security and resource constraints
**Rating:** 2 (Partial Support)

**Evidence:**

1. **Test Duration Constraints:**
   - Source: `mlperf.conf` lines 44-74
   - `*.SingleStream.min_duration = 600000` (milliseconds)
   - `*.Server.min_duration = 600000`
   - Configurable per scenario

2. **Rate Limiting (Implicit):**
   - Source: `mlperf.conf` lines 71-100
   - Server scenario: `*.Server.target_latency = 10` (ms)
   - QPS targets implicitly limit request rate
   - LoadGen controls query issuance rate

3. **Timeouts:**
   - Source: `language/deepseek-r1/utils/backend_registry.py` lines 114-115
   - SGLang: `request_timeout: None`, `server_startup_timeout: 1800`
   - vLLM: `VLLM_ENGINE_ITERATION_TIMEOUT_S: 0` (disabled)
   - Limited timeout configuration

4. **Resource Limits:**
   - Source: Backend configurations
   - GPU memory: `gpu_memory_utilization: 0.90`
   - Max sequences: `max_num_seqs: 64`
   - Max tokens: `max_tokens: 20000`

5. **Security Features:**
   - Docker containerization (isolation)
   - User management in launch_docker.sh
   - No explicit credential management or secrets handling
   - No sandboxing for code execution (e.g., LiveCodeBench uses temp dirs)

**Limitations:**
- **No API cost tracking** - Cannot track OpenAI/Anthropic API costs
- **No budget constraints** - Cannot set spending limits
- **No rate limit APIs** - LoadGen controls rate, but no per-API-key limits
- **Limited credential management** - Environment variables only
- **No execution sandboxing** for generated code (security risk)
- **No audit logging** of API calls or resource usage

**Why Partial Support:**
- ✓ Test duration limits
- ✓ Resource constraints (memory, batch size)
- ✓ Basic timeout support
- ✓ Docker isolation
- ✗ No cost tracking
- ✗ No API budget limits
- ✗ Limited credential management
- ✗ No code execution sandboxing

---

## Overall Assessment

**Strengths:**
1. **Excellent backend abstraction** - Clean multi-backend support (S1P4: 3/3)
2. **Well-defined benchmarks** - Standard datasets and metrics per task (S1P3: partial)
3. **Comprehensive test scenarios** - Multiple evaluation modes and scenarios (S1P2: partial)
4. **Production-grade LoadGen** - Battle-tested infrastructure for performance benchmarking

**Weaknesses:**
1. **Limited flexibility** - Fixed task-model-dataset triplets, hard to customize
2. **No advanced evaluation** - No LLM-as-judge, human-in-the-loop, or prompt-based evaluation (S1P5: 1/3)
3. **Minimal cost/security controls** - No API cost tracking, budget limits, or execution sandboxing (S1P6: partial)
4. **No unified abstractions** - Dataset loaders, evaluators, and configs are benchmark-specific

**Framework Type:**
MLPerf Inference is a **benchmark suite** rather than a general-purpose evaluation harness. It prioritizes:
- Standardized benchmarks for comparing systems
- Fair and reproducible performance measurement
- Strict adherence to MLPerf rules

It is **not designed** for:
- Rapid prototyping of custom evaluation tasks
- Flexible prompt engineering and few-shot evaluation
- LLM-as-judge or human evaluation workflows
- Cost-aware API evaluation

**Recommendations:**
- For MLPerf submission: Excellent (3/3)
- For custom task evaluation: Limited (1/3)
- For LLM research evaluation: Not recommended (1/3)
- For production performance benchmarking: Excellent (3/3)

**Comparison Context:**
- More structured than LM-Evaluation-Harness (which has flexible task configs)
- Less flexible than HELM (which has extensive evaluation scenarios)
- More focused on performance than DeepEval (which emphasizes LLM evaluation)
- Comparable to BigBench in benchmark standardization

---

## Evidence Summary

### Configuration Files Examined:
- `mlperf.conf`: Test settings, seeds, constraints
- `language/deepseek-r1/utils/backend_registry.py`: Backend configurations
- `loadgen/test_settings.h`: Test scenarios and modes

### Code Modules Examined:
- `language/deepseek-r1/backends/`: Backend implementations
- `language/deepseek-r1/eval_accuracy.py`: Accuracy evaluation
- `tools/submission/submission_checker.py`: Submission validation
- `loadgen/`: LoadGen C++ library

### Documentation Reviewed:
- `README.md`: Benchmark overview
- `loadgen/README.md`: LoadGen architecture
- `language/deepseek-r1/README.md`: DeepSeek-R1 benchmark guide

### Known Limitations (per documentation):
- LoadGen is NOT aware of models, datasets, or accuracy scoring (by design)
- Each benchmark requires custom implementation of SystemUnderTest and QuerySampleLibrary
- Accuracy evaluation scripts are benchmark-specific
- No modifications to LoadGen allowed for submissions (must upstream)
