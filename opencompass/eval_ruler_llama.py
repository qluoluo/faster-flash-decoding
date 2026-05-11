"""OpenCompass RULER eval config for FFD on Llama-3.1-8B-Instruct.

Usage:
    cd /path/to/opencompass
    opencompass ffd-core/opencompass/eval_ruler_llama.py -w output/ruler

Adjust MODEL_PATH below to point to your local checkpoint.
"""

from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from mmengine.config import read_base

# Import the FFD model wrapper (ensure ffd-core is installed and oc_model.py is reachable)
from opencompass.models.myModel.ffd.oc_model import HF_ForCausalLM_FFD_OC
from opencompass.models import HuggingFaceCausalLM_Strip as HuggingFaceCausalLM

with read_base():
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets

# ---- User settings ----
MODEL_PATH = "/path/to/Llama-3.1-8B-Instruct"
MAX_SEQ_LEN = 32 * 1024
MAX_OUT_LEN = 100
BATCH_SIZE = 1
RUN_CFG = dict(num_gpus=1, num_procs=1)

DEFAULT_MODEL_KWARGS = dict(
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="float16",
    attn_implementation="flash_attention_2",
)

datasets = list(ruler_datasets)

MODEL_CONFIG_LIST = [
    {"abbr": "ffd-delta5",    "delta": 5.0, "use_full_graph": True},
    {"abbr": "ffd-delta7",    "delta": 7.0, "use_full_graph": True},
]

models = []
for cfg in MODEL_CONFIG_LIST:
    abbr = cfg.pop("abbr")
    models.append(dict(
        type=HF_ForCausalLM_FFD_OC,
        abbr=abbr,
        path=MODEL_PATH,
        model_kwargs={**cfg, "use_sparse_decode": True, "BS": 128,
                      "use_fp8_residual": True, "k_bits": 2},
    ))

# Optional: dense baseline
models.append(dict(
    type=HuggingFaceCausalLM,
    abbr="base",
    path=MODEL_PATH,
    model_kwargs=DEFAULT_MODEL_KWARGS,
))

for m in models:
    m.setdefault("model_kwargs", {})
    m["model_kwargs"] = DEFAULT_MODEL_KWARGS | m["model_kwargs"]
    m.update(dict(
        tokenizer_path=MODEL_PATH,
        tokenizer_kwargs=dict(padding_side="left", truncation_side="left", trust_remote_code=True),
        max_seq_len=MAX_SEQ_LEN,
        max_out_len=MAX_OUT_LEN,
        run_cfg=RUN_CFG,
        batch_size=BATCH_SIZE,
    ))

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask), retry=1),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=160, task=dict(type=OpenICLEvalTask)),
)
