import json
import jsonlines
import queue
import numpy as np
import pandas as pd
from typing import Optional
from typer import Typer, Option
from copy import deepcopy
from functools import partial
from itertools import product
from pathlib import Path
from rich import print
from collections import defaultdict

from params import AllParams, ExperimentParams, LLMParams, sel2cls
from constants import Dataset as D, ExSel as ES, LMType as P, LLM
from tools.exp import get_strings, get_ints, get_datasets, get_exsels, get_lms
from data_params import ds2cls

app = Typer()
q = queue.Queue()

def process_params(
    params_l: list[AllParams], only_prompts: bool, only_incomplete: bool, preview: str,
    run: bool, paramsfile: str
):
    """Process the list of experiment parameters `params_l`.

    Args:
        params_l: list of experiment parameters
        only_prompts: only select demos and generate ICL prompts. (WIP)
        only_incomplete: only experiments that are not completed yet
        preview: just output a property of the experiment parameter (acceptable values: params, exp_path, commands, logfiles).
        run: dump all parameters to a jsonl file to be run using `run.run_exps_parallel`
        paramsfile: jsonl file to dump the parameters
    """
    print(f'Total {len(params_l)} experiments...')
    params_to_run: list[AllParams] = []
    for i, params in enumerate(params_l):
        if only_incomplete:
            if params.completed:
                print(f'Skipping experiment {i+1}/{len(params_l)}: {params.exp_path} ...')
                continue
        params_to_run.append(params)

    print(f'Running {len(params_to_run)} experiments...')
    if preview:
        for i, params in enumerate(params_to_run):
            if preview == 'params':
                print(f'\n{i+1}/{len(params_to_run)}:', params)
            elif preview == 'exp_path':
                print(f'\n{i+1}/{len(params_to_run)}:', params.exp_path)
            elif preview == 'commands':
                print(f'\n{i+1}/{len(params_to_run)}:', params.cmd)
            elif preview == 'logfiles':
                print(f'{i+1}/{len(params_to_run)}:', params.logfile if not only_prompts else params.promptslogfile)
            elif preview == 'outfiles':
                print(f'{i+1}/{len(params_to_run)}:', params.outfile if not only_prompts else params.promptsoutfile)
            else:
                print(f'Invalid preview option: {preview}')
    if run:
        with jsonlines.open(paramsfile, mode='w') as writer:
            # breakpoint()
            writer.write_all([p.to_dict() for p in params_to_run])

def compute_coverage_metrics(params_l: list[AllParams], progress=False):
    import shutil, json
    from tools.track import track
    from constants import Dataset as D
    from driver import get_templates
    from eval import get_substruct_fns, prompt_coverage

    all_substruct_fns = get_substruct_fns()
    coverage_metrics_l = []
    for params in track(params_l, disable=not progress):
        resultsfile = params.resultsfile
        if not resultsfile.exists():
            coverage_metrics_l.append(None)
            continue
        example_template = get_templates(
            params.dataset, params.prompt_format, params.input_feature, params.target_feature)['example_template']
        results = json.load(resultsfile.open())
        if params.dataset in [D.GEOQUERY, D.OVERNIGHT]:
            substruct_fns = all_substruct_fns
        else:
            substruct_fns = {k: v for k, v in all_substruct_fns.items() if 'lfst' not in k}
        coverage_metrics = {f'{k}_recall': 0 for k in substruct_fns.keys()}
        if 'coverage' in results and results['coverage'].keys() == coverage_metrics.keys():
            coverage_metrics_l.append(results['coverage'])
            continue
        for res in track(results['results']):
            if 'coverage' not in res:
                res['coverage'] = {}
            if res['coverage'].keys() != coverage_metrics.keys():
                missing_substruct_fns = {k:v for k, v in substruct_fns.items() if f'{k}_recall' not in res['coverage']}
                if missing_substruct_fns:
                    coverage = prompt_coverage(
                        res, missing_substruct_fns, example_template, params.n_shots)
                    res['coverage'] |= coverage
            for k, v in res['coverage'].items():
                coverage_metrics[k] += v / len(results['results'])
        results['coverage'] = coverage_metrics
        coverage_metrics_l.append(coverage_metrics)
        shutil.move(resultsfile, f'{resultsfile}.bak.2')
        json.dump(results, resultsfile.open('w'), indent=2)
    return coverage_metrics_l

def get_single_results(i, N, P: AllParams, coverage_results=False, verbose=True):
    """load results for a single experiment"""

    # finalresults = P.to_flattened_dict()
    finalresults = {k.split('.')[-1]: v for k, v in P.to_flattened_dict().items()}
    finalresults['completed'] = P.completed
    if P.selector.selector_type not in [ES.RANDOM]:
        finalresults |= dict(selector_name=P.selector_name)
    logfile = P.logfile
    resultsfile = P.resultsfile
    lastlog = ''
    if logfile.exists():
        try:
            lines = open(logfile).readlines()
            if lines: lastlog = lines[-1].strip()
        except:
            pass
    # print(f'{i+1}/{N}', resultsfile, resultsfile.exists(), P.promptsfile.exists(), lastlog)
    if verbose: print(f'{i+1}/{N}', resultsfile, resultsfile.exists(), lastlog)
    if resultsfile.exists():
        finalresults |= dict(completed=True)
        results = json.load(resultsfile.open())
        metrics = results['metrics']
        if 'result' not in metrics:
            if P.dataset not in [D.BREAK, D.COMMONGEN, D.E2ENLG, D.DART]:
                metrics['result'] = metrics['accuracy']
            elif P.dataset == D.BREAK:
                metrics['result'] = metrics['lfem']
            else:
                metrics['result'] = metrics['rougeL']

        # for measuring coverage with the demonstrations -- not used currently
        coverage_metrics = ['ngram_1_recall', 'ngram_4_recall', 'depst_4_recall', 'lfst_4_recall']

        # selection and ICL time
        time_metrics = ['sel_time', 'icl_time']

        # different datasets will have different ICL evaluation metrics.
        # 'result' will be populated by the main metric for all of them. See `data_params.py` for details.
        icl_eval_metric_cols = ['bleu', 'lfem', 'avg_n_shots', 'rougeL', 'result']

        for m in coverage_metrics + time_metrics + icl_eval_metric_cols:
            if m in metrics: finalresults |= {m: metrics[m]}
        if 'n_shots' in metrics:
            finalresults |= dict(avg_n_shots=metrics['n_shots'])
        if P.n_shots == 1000:
            assert P.promptsfile.exists(), P.promptsfile
        if P.promptsfile.exists():
            promptsdata = json.load(P.promptsfile.open())
            finalresults['sel_time'] = promptsdata['metrics']['sel_time']
            # classification accuracy if taking the majority vote of the demonstrations
            if 'majority_result' in promptsdata['metrics']:
                finalresults['majority_result'] = promptsdata['metrics']['majority_result']
                finalresults['majority_precision'] = promptsdata['metrics']['majority_precision']
                from collections import Counter
                finalresults['majority_freq'] = Counter(
                    max(map(r['demo_targets'].count, set(r['demo_targets'])))
                    for r in promptsdata['results'])
        if P.selector_type == ES.GIST_BERTSCORE:
            gistlm_evalfile = Path(P.selector.emb_lm) / 'eval/eval_results.json'
            if gistlm_evalfile.exists():
                # print(gistlm_evalfile)
                gistlm_eval = json.load(gistlm_evalfile.open())
                if P.dataset == D.SMCALFLOW_CS: split = P.split
                elif P.dataset == D.COGS: split = P.test_split
                else: split = 'validation'
                finalresults['glm_result'] = 100 * gistlm_eval[f'{split}_eval_accuracy']
        if coverage_results:
            if not 'coverage' in metrics:
                print('Computing coverage metrics...')
                metrics['coverage'] = compute_coverage_metrics([P])[0]
            finalresults |= metrics['coverage']
    return finalresults

def get_results(params_l: list[AllParams], coverage_results=False, parallel=False, verbose=True):
    """load results for all experiments in `params_l`"""
    from joblib import Parallel, delayed
    if parallel:
        with Parallel(n_jobs=20, verbose=True) as parallel:
            results_l = parallel(delayed(get_single_results)(
                    i, len(params_l), params, coverage_results, verbose)
                for i, params in enumerate(params_l))
    else:
        results_l = [get_single_results(i, len(params_l), params, coverage_results, verbose)
                    for i, params in enumerate(params_l)]
    if not results_l: return None
    resultsdf = pd.DataFrame.from_records(results_l)
    return resultsdf

def postprocess_params(params: AllParams):
    EP, DP, LP, SP = params.shorthand

    if EP.tiny: # "tiny" run for debugging
        DP.n_cands, DP.n_test = 40, 20

    if SP.selector_type == ES.GIST_BERTSCORE:   # always prefix with task description if available for gist LMs
        from params import GistBertScoreSelectorArgs
        SP: GistBertScoreSelectorArgs
        if 'flan_' in SP.emb_lm:
            DP.prefix = True
    return params

@app.command()
def main(
    label: str = 'exp0',    # label for experiment (see params.AllParams).
    debug: bool = False,    # run the experiment in debug mode.
    tiny: bool = False,     # set n_cands and n_test in each AllParams to a small value.
    data_root: Path = Path('../data'),  # directory containing local datasets (see params.AllParams).
    output_root: Path = Path('../results'), # Directory to write logs and experimental results to.
    gistlms_root: Path = Path('../../gistlms/'),    # Directory containing Gist LMs.
    datasets: str = 'GEOQUERY', # list of names from `constants.Dataset`` as a ';' separated string
    lms: str = 'neo;llama-7B', # list of LLMs as a ';' separated string. See `exp_utils.lm_args_d` for acceptable LM names.
    selectors: str = 'random',  # list of selectors as a ';' separated string. See `exp_utils.selector_args_d` for acceptable selector names.
    seeds: str = '0',   # list of seeds as a ';' separated string.
    n_shots: Optional[str] = None,  # override the number of shots to try. list of integers as a ';' separated string.
    n_cands: Optional[str] = None,  # override the number of candidates. list of integers as a ';' separated string.
    batch_sizes: Optional[str] = None,  # `AllParams.batch_size` and `AllParams.lm.lm_batch_size` in a ';' separated string. Used to override the default.
    extra: Optional[str] = None,    # Additional overrides.
    baselines_exp: Optional[bool] = False,
    return_params: bool = False,    # return the list of parameters and exit.
    only_incomplete: bool = False,  # filter the experiments that have already finished.
    only_prompts: bool = False, # used in `process_params`
    preview: str | None = None, # used in `process_params`
    run: bool = False,  # used in `process_params`
    paramsfile: Path = Path('params.jsonl'),    # used in `process_params`
    collate_results: bool = True,   # whether to collate results.
    collate_results_file: Optional[str] = None,  # where to save collated results
    coverage_results: bool = False, # not used
    verbose: bool = True,   # whether to log during `get_results`
    bs_multiplier: int = 1,
    only_1tok: bool = False,
    only_large: bool = True,
    balance: bool = False
):
    """
    Top-level command for running/collating results for batches of experiments.
    It will create a list of experiment parameters (objects of params.AllParams)
    based on the cross-product of `datasets`, `lms`, `selectors` and `seeds`
    and either return the list (if `return_params=True`) or process them
    using `experiments.process_params` which will log different properties
    of each AllParams based on `preview` and dump them in `paramsfile` to be run
    using `run.run_exps_parallel`.
    Finally, if `collate_results=True`, it will collect the results for all the experiments
    using `get_results` and collate all the parameters and metrics in a pandas table.

    Example Usage:
    ```bash
    python experiments.py --label "exp2" --seeds 0 --datasets "QNLI;MNLI;RTE;SST2;YELP;MRPC;QQP;PAWS;COPA;PIQA;WINOGRANDE;WSC;CMSQA;COLA;COMMONGEN;E2ENLG;DART;SST5;AGNEWS;AESLC;SMCALFLOW_CS;BREAK;MTOP;COGS" --selectors "cosine;bertscore;gist_bertscore" --n-shots 8 --baselines-exp --paramsfile "params/all.jsonl" --run --preview "logfiles"
    ```
    """
    overrides = dict(exp={}, data={}, llm={}, selector={})
    if n_shots: overrides['selector']['n_shots'] = get_ints(n_shots)
    if n_cands: overrides['data']['n_cands'] = get_ints(n_cands)
    if extra:
        # parse override assignments from extra as a dictionary
        for override in extra.split(';'):
            key, value = override.split('=')
            sub, param = key.split('.')
            overrides[sub][param] = value
    from exp_utils import ds2splits, dataset_args_d, lm_args_d, selector_args_d, geoquery_splits, \
    cosine_emb_lms, bertscore_emb_lms, ds2gistlms, multitask_pretrained_gistlms

    def get_params_l(
        seed, dataset, split, lm, selector, n_cands=-1, batch_sizes=None, selector_args={}):
        lmds2bs = defaultdict(lambda: None, {})
        if dataset not in [D.YELP, D.PIQA, D.RTE, D.AESLC, D.AGNEWS, D.DART, D.DROP, D.BOOLQ]:
            lm2bs = defaultdict(lambda: '28;7', {'neo': '24;6', 'davinci': '80;20', 'llama-7B': '24;2', 'mistral': '24;2', 'zephyr': '24;2', 'llama-13B': '24;1', 'starcoder': '24;1', 'turbo': '24;1', 'turbo-june': '24;1'})
        else:
            lm2bs = defaultdict(lambda: '28;7', {'neo': '20;4', 'davinci': '80;10', 'llama-7B': '24;1', 'mistral': '24;1', 'zephyr': '24;1', 'llama-13B': '24;1', 'starcoder': '24;1', 'turbo': '24;1', 'turbo-june': '24;1'})
        batch_sizes = batch_sizes or lmds2bs[(lm, dataset)] or lm2bs[lm]
        batch_size, lm_batch_size = get_ints(batch_sizes)
        lm_batch_size *= bs_multiplier

        exp_args = dict(
            label=label, data_root=data_root, output_root=output_root,
            debug=debug, tiny=tiny, only_prompts=only_prompts,
            batch_size=batch_size, seed=seed, balance=balance) | overrides['exp']
        dataset_args = dataset_args_d.get(dataset, dict()) | dict(
            prefix=True, n_cands=n_cands, n_test=-1, split=split,
        ) | overrides['data']
        # if lm.startswith('turbo') or lm == 'davinci' or lm == 'davinci-002':
        #     dataset_args['n_test'] = 250
        lm_args = lm_args_d[lm] | dict(lm_batch_size=lm_batch_size) | overrides['llm']
        selector_args = selector_args_d[selector] | selector_args | overrides['selector']
        selector_type = selector_args['selector_type']

        return AllParams(
            exp=ExperimentParams(**exp_args),
            data=ds2cls[dataset](**dataset_args),
            llm=LLMParams(**lm_args),
            selector=sel2cls[selector_type](**selector_args)
        ).get_settings()

    params_l: list[AllParams] = []
    for seed, dataset, lm, selector in product(
        get_ints(seeds), get_datasets(datasets), get_strings(lms), get_strings(selectors)
    ):
        splits = ds2splits.get(dataset, [None])
        if selector == 'lf_coverage':
            if dataset == D.SMCALFLOW_CS: splits = ['8_S', '32_C']
            elif dataset == D.GEOQUERY: splits = geoquery_splits
            else: continue
        # if lm in ['babbage', 'davinci-002'] and selector in ['random', 'bm25']:
        #     continue

        # splits = ds2splits.get(dataset, None)
        # n_cands = -1 if dataset not in [D.MNLI] else 44000
        n_cands = 44000
        for split in splits:
            if dataset == D.GEOQUERY and seed > 0 and 'lf' not in selector \
                    and split is not None and ('csl_template' in split or 'csl_tmcd' in split):
                continue
            common = [seed, dataset, split, lm, selector, n_cands, batch_sizes]
            get_params_fn = partial(get_params_l, *common)
            if selector == 'random':
                params_l += get_params_fn()
            elif selector in ['cosine', 'cosine_coverage']:
                emb_lms = 'sentence-transformers/all-mpnet-base-v2' if baselines_exp else cosine_emb_lms
                params_l += get_params_fn(selector_args=dict(emb_lm=emb_lms))
            elif selector == 'bertscore':
                idfs = False if baselines_exp else [True, False]
                emb_lms = 'microsoft/deberta-large-mnli' if baselines_exp else bertscore_emb_lms
                params_l += get_params_fn(selector_args=dict(idf=idfs, emb_lm=emb_lms))
            elif selector == 'bertscore_prec':
                emb_lms = 'microsoft/deberta-large-mnli' if baselines_exp else bertscore_emb_lms
                params_l += get_params_fn(selector_args=dict(idf=False, emb_lm=emb_lms))
            elif selector == 'set_bsr':
                idfs = False if baselines_exp else [True, False]
                emb_lms = 'microsoft/deberta-large-mnli' if baselines_exp else bertscore_emb_lms
                orderings = 'recall' if baselines_exp else [None, 'recall']
                params_l += get_params_fn(selector_args=dict(
                    idf=idfs, emb_lm=emb_lms, ordering=orderings))
            elif selector == 'gist_bertscore':
                layers = -1 if baselines_exp else [-1, 23, 22, 21, 20]
                gistlms = ds2gistlms(dataset, split, gistlms_root)
                if only_large:
                    gistlms = [glm for glm in gistlms if not 'pretrains' in glm or 'large' in glm]
                if only_1tok:
                    gistlms = [glm for glm in gistlms if '1tok' in glm]
                elif lm in ['davinci-002', 'babbage']:
                    gistlms = [glm for glm in gistlms if '1tok' in glm or '3tok' in glm]
                if len(gistlms) == 0: continue
                params_l += get_params_fn(selector_args=dict(
                    dataset=dataset, split=split, layer=layers, emb_lm=gistlms))
            elif selector == 'set_gbsr':
                gistlms = ds2gistlms(dataset, split, gistlms_root)
                # gistlms = [glm for glm in gistlms if '3tok' in glm]
                if only_large:
                    gistlms = [glm for glm in gistlms if not 'pretrains' in glm or 'large' in glm]
                if only_1tok:
                    gistlms = [glm for glm in gistlms if '1tok' in glm]
                elif lm in ['davinci-002', 'babbage']:
                    gistlms = [glm for glm in gistlms if '1tok' in glm or '3tok' in glm]
                if len(gistlms) == 0: continue
                params_l += get_params_fn(selector_args=dict(
                    dataset=dataset, split=split, emb_lm=gistlms, idf=False, ordering='recall'))
            elif selector in ['recall', 'bm25', 'recall_coverage', 'bm25_coverage', 'bm25_coverage_candscore']:
                if baselines_exp:
                    if selector == 'bm25': params_l += get_params_fn(
                        selector_args=dict(substruct='ngram', subst_size=1))
                    elif selector == 'bm25_coverage': params_l += get_params_fn(
                        selector_args=dict(substruct='ngram', ordering='bm25', subst_size=4))
                    else: continue
                else:
                    if dataset != D.NL2BASH: params_l += get_params_fn(
                        selector_args=dict(substruct='depst', subst_size=4))
                    params_l += get_params_fn(selector_args=dict(substruct='ngram', subst_size=[1, 4]))
            elif selector == 'lf_coverage':
                params_l += [
                    *get_params_l(0, *common),
                    *get_params_l(1, *common),
                    *get_params_l(2, *common),
                ]

    params_l = [postprocess_params(p) for p in params_l]
    from collections import Counter
    freqs = Counter([p.exp_path for p in params_l])
    if freqs.most_common(1)[0][1] > 1:
        print('WARNING: duplicate params')
        print(freqs.most_common(5))

    if return_params: return params_l

    process_params(params_l, only_prompts, only_incomplete, preview, run, paramsfile)
    if collate_results:
        resultsdf: pd.DataFrame = get_results(params_l, coverage_results=coverage_results, verbose=verbose)
        if resultsdf is not None and resultsdf.completed.any() and verbose:
            make_tables(resultsdf, resultsfile=collate_results_file, aggregate_csl=False)
        return resultsdf


def make_tables(
    resultsdf: pd.DataFrame, output=True, resultsfile=None,
    aggregate_csl=False, count=True, fillna=True, summarize=True
) -> pd.DataFrame:
    """Do some aggregation and post-processing on the all the results."""
    resultsdf = resultsdf[resultsdf.completed]
    filter_cols = lambda cols, allowed: [c for c in cols if c in allowed]

    common_cols = ['dataset', 'input_feature', 'split', 'test_split', 'n_test', 'n_cands',
                   'sel_prompt_version', 'icl_prompt_version', 'prompt_format',
                   'n_shots', 'lm_name', 'selector_type', 'selector_name']
    similar_cols = ['emb_lm', 'sim_metric']
    struct_cols = ['substruct', 'subst_size', 'ordering', 'selector_metric', 'coverage',]
    coverage_cols = ['n_combs', 'greedy_coverage', 'depparser',
                     'template_diversity', 'use_paraphrase', 'break_on_reset']
    bertscore_cols = ['idf', 'embed_context', 'layer']
    cosine_cols = ['reorder']
    columns = filter_cols(
        [*common_cols, *similar_cols, *struct_cols, *coverage_cols, *bertscore_cols, *cosine_cols],
        resultsdf.columns)
    resultsdf = resultsdf.sort_values(columns)

    # geoquery has multiple csl splits of each type (eg. csl_tmcd_{1,2,3}) with different seeds.
    # sepearate out the "csl_seed" so it will be aggregated later similar to `seed`.
    if aggregate_csl:
        def csl(row):   # eg. csl_tmcd_1 --> csl_tmcd, 1
            if row.split is not None and (row.split.startswith('csl_tmcd') or row.split.startswith('csl_template')):
                parts = row.split.split('_')
                return pd.Series(['_'.join(parts[:2]), parts[2]])
            else:
                return pd.Series([row.split, None])
        resultsdf[['split', 'csl_seed']] = resultsdf.apply(csl, axis=1)

    # -1 is used for n_cands and n_test to indicate all train and test instances respectively
    resultsdf[columns] = resultsdf[columns].replace(-1, 'all', regex=True)

    # cleanup columns that have the same value for all experiments
    for col in columns:
        if col not in ['dataset', 'split'] and len([v for v in resultsdf[col].unique() if v is not None]) <= 1:
            resultsdf = resultsdf.drop(col, axis=1)


    # for measuring coverage with the demonstrations -- not used currently
    coverage_cols = ['ngram_1_recall', 'ngram_4_recall', 'depst_4_recall', 'lfst_4_recall']
    # selection and ICL time
    time_cols = ['sel_time', 'icl_time']
    # different datasets will have different ICL evaluation metrics.
    # 'result' will be populated by the main metric for all of them. See `data_params.py` for details.
    icl_eval_metric_cols = ['bleu', 'lfem', 'avg_n_shots', 'rougeL', 'result']

    # final aggregated dataframe
    metric_cols = filter_cols(coverage_cols + time_cols + icl_eval_metric_cols + ['majority_result', 'majority_precision', 'majority_freq'][:-1] + ['glm_result'], resultsdf.columns)
    final_cols = filter_cols(columns, resultsdf.columns)
    final_df: pd.DataFrame = resultsdf.groupby(final_cols, dropna=False).agg({
        'result': ['mean', 'count'] if count else 'mean',
        **{c: 'mean' for c in metric_cols if c not in ['result', 'majority_freq']},
        # 'majority_freq': 'first',
    })

    # fill nans in index with '-'
    if fillna:
        def fillna_index(df, my_fillna_value):
            if isinstance(df.index, pd.MultiIndex):
                df.index = pd.MultiIndex.from_frame(
                    df.index.to_frame().fillna(my_fillna_value)
                )
            else:
                df.index = df.index.fillna(my_fillna_value)
            return df

        final_df = fillna_index(final_df, '-')

    # output the final df and a further summarized df
    if output:
        with pd.option_context(
            'display.max_rows', None,
            # 'display.max_columns', None,
            "display.max_colwidth", 70,
            'display.precision', 2,
        ):
            print(final_df)
            if summarize:
                summary_df = final_df.reset_index().droplevel(level=1, axis=1)
                params_cols = filter_cols(
                    ['dataset', 'split', 'test_split', 'icl_prompt_version', 'sel_prompt_version', 'lm_name', 'selector_type', 'selector_name', 'layer', 'coverage'],
                    summary_df.columns)
                metric_cols = filter_cols(['lfem', 'rougeL', 'result'], summary_df.columns)
                if params_cols and metric_cols:
                    summary_df = summary_df[[*params_cols, *metric_cols]].set_index(params_cols)
                    print(summary_df)
                    # print(final_df.reset_index()[summary_cols].set_index(summary_cols[:-2]))

    # dump to file
    if resultsfile:
        final_df.to_latex(f'{resultsfile}.tex', escape=False, multirow=True, multicolumn=True, float_format='%.2f', column_format='l' + 'r' * (len(final_df.columns) - 1))
        final_df.to_excel(f'{resultsfile}.xlsx', index=True, merge_cells=True, float_format='%.2f')

    return final_df

if __name__ == '__main__':
    app()