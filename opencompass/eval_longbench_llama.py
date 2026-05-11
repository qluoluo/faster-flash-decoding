"""OpenCompass LongBench eval config for FFD on Llama-3.1-8B-Instruct.

Usage:
    cd /path/to/opencompass
    opencompass ffd-core/opencompass/eval_longbench_llama.py -w output/longbench

Adjust MODEL_PATH below to point to your local checkpoint.
"""

from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from mmengine.config import read_base

from opencompass.models.myModel.ffd.oc_model import HF_ForCausalLM_FFD_OC
from opencompass.models import HuggingFaceCausalLM_Strip as HuggingFaceCausalLM

with read_base():
    from opencompass.configs.datasets.longbench.longbenchnarrativeqa.longbench_narrativeqa_gen import LongBench_narrativeqa_datasets
    from opencompass.configs.datasets.longbench.longbenchqasper.longbench_qasper_gen import LongBench_qasper_datasets
    from opencompass.configs.datasets.longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen import LongBench_multifieldqa_en_datasets
    from opencompass.configs.datasets.longbench.longbenchmultifieldqa_zh.longbench_multifieldqa_zh_gen import LongBench_multifieldqa_zh_datasets
    from opencompass.configs.datasets.longbench.longbenchhotpotqa.longbench_hotpotqa_gen import LongBench_hotpotqa_datasets
    from opencompass.configs.datasets.longbench.longbench2wikimqa.longbench_2wikimqa_gen import LongBench_2wikimqa_datasets
    from opencompass.configs.datasets.longbench.longbenchmusique.longbench_musique_gen import LongBench_musique_datasets
    from opencompass.configs.datasets.longbench.longbenchdureader.longbench_dureader_gen import LongBench_dureader_datasets
    from opencompass.configs.datasets.longbench.longbenchgov_report.longbench_gov_report_gen import LongBench_gov_report_datasets
    from opencompass.configs.datasets.longbench.longbenchqmsum.longbench_qmsum_gen import LongBench_qmsum_datasets
    from opencompass.configs.datasets.longbench.longbenchmulti_news.longbench_multi_news_gen import LongBench_multi_news_datasets
    from opencompass.configs.datasets.longbench.longbenchvcsum.longbench_vcsum_gen import LongBench_vcsum_datasets
    from opencompass.configs.datasets.longbench.longbenchtrec.longbench_trec_gen import LongBench_trec_datasets
    from opencompass.configs.datasets.longbench.longbenchtriviaqa.longbench_triviaqa_gen import LongBench_triviaqa_datasets
    from opencompass.configs.datasets.longbench.longbenchsamsum.longbench_samsum_gen import LongBench_samsum_datasets
    from opencompass.configs.datasets.longbench.longbenchlsht.longbench_lsht_gen import LongBench_lsht_datasets
    from opencompass.configs.datasets.longbench.longbenchpassage_count.longbench_passage_count_gen import LongBench_passage_count_datasets
    from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_en.longbench_passage_retrieval_en_gen import LongBench_passage_retrieval_en_datasets
    from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_zh.longbench_passage_retrieval_zh_gen import LongBench_passage_retrieval_zh_datasets
    from opencompass.configs.datasets.longbench.longbenchlcc.longbench_lcc_gen import LongBench_lcc_datasets
    from opencompass.configs.datasets.longbench.longbenchrepobench.longbench_repobench_gen import LongBench_repobench_datasets

datasets = sum((v for k, v in locals().items() if k.endswith("_datasets")), [])

# ---- User settings ----
MODEL_PATH = "/path/to/Llama-3.1-8B-Instruct"
INPUT_MAX_SEQ_LEN = 31500
CACHE_MAX_SEQ_LEN = 32 * 1024
MAX_OUT_LEN = 64
BATCH_SIZE = 1
RUN_CFG = dict(num_gpus=1, num_procs=1)

DEFAULT_MODEL_KWARGS = dict(
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="float16",
    attn_implementation="flash_attention_2",
)

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
                      "cache_max_seq_len": CACHE_MAX_SEQ_LEN,
                      "use_fp8_residual": True, "k_bits": 2},
    ))

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
        max_out_len=MAX_OUT_LEN,
        run_cfg=RUN_CFG,
        batch_size=BATCH_SIZE,
        drop_middle=True,
        max_seq_len=INPUT_MAX_SEQ_LEN,
    ))

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask), retry=1),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=160, task=dict(type=OpenICLEvalTask)),
)
