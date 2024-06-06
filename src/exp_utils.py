from pathlib import Path
from constants import Dataset as D, LLM, ExSel as ES, LMType as P

lm_args_d = {
    'openai': dict(
        lm_type=P.OPENAI, lm_url=None,
        lm_name=[LLM.CODE_CUSHMAN_001, LLM.CODE_DAVINCI_002],
        lm_batch_size=7, lm_delay=10,),
    'cushman': dict(
        lm_name=LLM.CODE_CUSHMAN_001, lm_type=P.OPENAI, lm_url=None,
        lm_batch_size=7, lm_delay=2, openai_keys_file='../../codex_keys.txt'),
    'codex': dict(
        lm_name=LLM.CODE_DAVINCI_002, lm_type=P.OPENAI, lm_url=None,
        lm_batch_size=7, lm_delay=2, openai_keys_file='../../codex_keys.txt'),
    'turbo': dict(
        lm_name=LLM.TURBO, lm_type=P.OPENAI_CHAT, lm_url=None,
        lm_batch_size=1, lm_delay=10,),
    'turbo-june': dict(
        lm_name=LLM.TURBO_JUNE, lm_type=P.OPENAI_CHAT, lm_url=None,
        lm_batch_size=1, lm_delay=10,),
    # 'davinci': dict(
    #     lm_name=LLM.TEXT_DAVINCI_003, lm_type=P.OPENAI, lm_url=None,
    #     lm_batch_size=7, lm_delay=1,),
    'davinci': dict(
        lm_name=LLM.TEXT_DAVINCI_002, lm_type=P.OPENAI, lm_url=None,
        lm_batch_size=7, lm_delay=1,),
    'babbage': dict(
        lm_name=LLM.BABBAGE_002, lm_type=P.OPENAI, lm_url=None,
        lm_batch_size=7, lm_delay=1,),
    'davinci-002': dict(
        lm_name=LLM.DAVINCI_002, lm_type=P.OPENAI, lm_url=None,
        lm_batch_size=7, lm_delay=1,),
    # 'opt': dict(
    #     lm_name=LLM.OPT_30B, lm_type=P.OPT_SERVER, lm_batch_size=7, lm_delay=10,
    #     lm_url='http://ava-s1.ics.uci.edu:8890',),
    # 'jt6b': dict(
    #     lm_name=LLM.JT6B, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=7,),
    # 'neox': dict(
    #     lm_name=LLM.NEOX20B, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=2,),
    'neo': dict(
        lm_name=LLM.NEO, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=10,),
    'llama-7B': dict(
        lm_name=LLM.LLAMA7B, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=7,),
    'llama-13B': dict(
        lm_name=LLM.LLAMA13B, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=7,),
    'starcoder': dict(
        lm_name=LLM.STARCODER, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=7,),
    'mistral': dict(
        lm_name=LLM.MISTRAL, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=7,),
    'zephyr': dict(
        lm_name=LLM.ZEPHYR, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=7,),
    'dolly3b': dict(
        lm_name=LLM.DOLLY3B, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=7,),
    'dolly7b': dict(
        lm_name=LLM.DOLLY7B, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=7,),
}

dataset_args_d: dict[D, dict] = {
    D.SMCALFLOW_CS: dict(
        input_feature='source',
        target_feature='target',
        sel_prompt_version=['v0', 'v1'][-1],
    ),
    D.COGS: dict(
        test_split=['gen', 'dev'],
        sel_prompt_version=['v0', 'v1'][-1],
    ),
    D.MTOP: dict(sel_prompt_version=['v0', 'v1'][-1],),
    D.SST5: dict(icl_prompt_version=['v0', 'v1'][0],),
    D.GSM8K: dict(prefix=True, sel_prompt_version=['v0', 'v1', 'v2'][0],),
}

selector_args_d: dict[str, tuple[ES, dict]] = {
    'random': dict(selector_type=ES.RANDOM),
    'cosine': dict(selector_type=ES.COSINE, coverage=False),
    'cosine_coverage': dict(selector_type=ES.COSINE, coverage=True, reorder=[False, True]),
    'recall': dict(selector_type=ES.STRUCT, metric='recall', coverage=False),
    'recall_coverage': dict(selector_type=ES.STRUCT,
        metric='recall', coverage=True, ordering=[None, 'recall'][1]),
    'bm25': dict(selector_type=ES.STRUCT, metric='bm25', coverage=False),
    'bm25_coverage': dict(selector_type=ES.STRUCT,
        metric='bm25', coverage=True, ordering=[None, 'bm25'], add_cand_score=False),
    'bm25_coverage_candscore': dict(selector_type=ES.STRUCT,
        metric='bm25', coverage=True, ordering=[None, 'bm25'], add_cand_score=True,
        cand_score_discount=[1, 3]),
    'bertscore': dict(selector_type=ES.BERTSCORE, metric='recall', coverage=False),
    'bertscore_prec': dict(selector_type=ES.BERTSCORE, metric=['precision', 'f1'], coverage=False),
    'set_bsr': dict(selector_type=ES.BERTSCORE,
        metric='recall', coverage=True, add_cand_score=[False, True][:1]),
    'lf_coverage': dict(selector_type=ES.LF_COVERAGE),
    'gist_bertscore': dict(selector_type=ES.GIST_BERTSCORE, metric='recall', coverage=False),
    'set_gbsr': dict(selector_type=ES.GIST_BERTSCORE,
        metric='recall', coverage=True),
}

geoquery_splits = [
    'iid', 'csl_length',
    *[f'csl_template_{i}' for i in range(1, 4)],
    *[f'csl_tmcd_{i}' for i in range(1, 4)],
]

ds2splits = {
    D.OVERNIGHT: ['socialnetwork_iid_0', 'socialnetwork_template_0'],
    D.ATIS: ['iid_0', 'template_0'],
    D.GEOQUERY: geoquery_splits,
    # D.SMCALFLOW_CS: ['0_S', '8_S', '0_C', '8_C', '16_C', '32_C'],
    D.SMCALFLOW_CS: ['8_S', '32_C'],
    # D.PAWSX: ['fr', 'es', 'de', 'zh'][:-1],
    # D.XNLI: ['fr', 'de', 'ru'],
    D.PAWSX: ['fr', 'es'],
    D.XNLI: ['de', 'ru'],
    D.TWEET: ['emotion', 'offensive', 'irony', 'stance'][:2],
    D.CFQ: ['mcd1', 'random_split'],
}

cosine_emb_lms = ['bert-base-uncased', 'sentence-transformers/all-mpnet-base-v2']
bertscore_emb_lms = ['microsoft/deberta-base-mnli', 'microsoft/deberta-large-mnli', 't5-large']

finetuned_gistlm = 'finetunes/{dataset}/v3-{finetune_name}-{n_tok}tok-flan-t5-{size}'
finetuned_gistlm_with_split = 'finetunes/{dataset}/{split}/v3-{finetune_name}-{n_tok}tok-flan-t5-{size}'
pretrained_gistlm = 'pretrains/flan2022_zs_len256_max10K-{n_tok}tok-flan-t5-{size}/checkpoint-{ckpt}'
multitask_pretrained_gistlms = [
    pretrained_gistlm.format(n_tok=1, size='large', ckpt=44000),
    pretrained_gistlm.format(n_tok=3, size='large', ckpt=46000),
    pretrained_gistlm.format(n_tok=6, size='large', ckpt=44000),
    pretrained_gistlm.format(n_tok=15, size='large',  ckpt=42000),

    pretrained_gistlm.format(n_tok=1, size='xl', ckpt=33000),
    pretrained_gistlm.format(n_tok=3, size='xl', ckpt=33500),
    pretrained_gistlm.format(n_tok=6, size='xl', ckpt=29500),
    pretrained_gistlm.format(n_tok=10, size='xl',  ckpt=29000),
]
print(multitask_pretrained_gistlms)
def ds2gistlms(ds: D, split: str = None, gistlms_root: Path = '../../gistlms'):
    gistlms = []
    gistlms += multitask_pretrained_gistlms
    if ds in [D.MTOP, D.SMCALFLOW_CS, D.COGS, D.SPIDER]:
        gistlms += [
            finetuned_gistlm.format(
                dataset=ds.name, finetune_name='vanilla', size='base', n_tok=k)
            for k in [1, 3, 6, 15]]
    elif ds in [D.CFQ]:
        gistlms += [
            finetuned_gistlm_with_split.format(
                dataset=ds.name, split=split, finetune_name='vanilla', size='base', n_tok=k)
            for k in [1, 3, 6, 15]]
    elif split is None:
        gistlms += [
            finetuned_gistlm.format(
                dataset=ds.name, finetune_name='vanilla', size='base', n_tok=k)
            for k in [1, 3]]
    else:
        gistlms += [
            finetuned_gistlm_with_split.format(
                dataset=ds.name, split=split, finetune_name='vanilla', size='base', n_tok=k)
            for k in [1, 3]]

    gistlms_paths: list[Path] = [Path(gistlms_root) / lm for lm in gistlms]
    # print(gistlms_paths)
    def glm_exists(glm_path: Path):
        if glm_path.exists():
            for f in glm_path.iterdir():
                if f.name.startswith('pytorch_model') or f.name == 'model.safetensors':
                    return True
        return False
    return [str(glm_path) for glm_path in gistlms_paths if glm_exists(glm_path)]
