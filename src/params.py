from __future__ import annotations
import hydra
import attr

from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from typing import Optional
from pathlib import Path
from enum import Enum

from constants import Dataset as D, ExSel as ES, LMType as P, LLM
from selector import (
    CommonSelectorArgs,
    CosineCoverageSelectorArgs,
    BertScoreSelectorArgs,
    GistBertScoreSelectorArgs,
    StructuralCoverageSelectorArgs,
    LFCoverageSelectorArgs,
)
from data_params import DataParams, ds2cls
from tools.param_impl import Parameters, converter

sel2cls: dict[ES, type] = {
    ES.RANDOM: CommonSelectorArgs,
    ES.COSINE: CosineCoverageSelectorArgs,
    ES.BERTSCORE: BertScoreSelectorArgs,
    ES.GIST_BERTSCORE: GistBertScoreSelectorArgs,
    ES.STRUCT: StructuralCoverageSelectorArgs,
    ES.LF_COVERAGE: LFCoverageSelectorArgs,
}
num2str = lambda x: 'all' if x == -1 else str(x)

@attr.s(auto_attribs=True)
class ExperimentParams(Parameters):
    label: str = 'exp0'                 # Experiment label.
    data_root: Path = '../data'         # Path to the data root.
    output_root: Path = '../results'    # Path to the output root.
    debug: bool = False                 # Debug mode.
    tiny: bool = False                  # Tiny mode.
    only_prompts: bool = False          # Generate prompts then exit.
    seed: int = 0                       # Random seed.
    gpu: int = 0                        # GPU to use.
    batch_size: Optional[int] = 20      # batch size aggregating ICL results in eval.py
    balance: bool = False               # Whether to use balanced selection of examples
    
@attr.s(auto_attribs=True)
class LLMParams(Parameters):
    """Common class for parameters for all LLM types."""
    lm_type: P = P.OPENAI               # LLM type.
    lm_name: LLM = LLM.CODE_DAVINCI_002   # LLM name.
    lm_url: Optional[str] = 'http://ava-s2.ics.uci.edu:8890' # LLM URL.
    openai_keys_file: str = '../openai_keys.txt' # Path to the OpenAI keys file.
    do_sample: bool = False             # Whether to sample from the LLM.
    temperature: float = 0.0            # Sampling temperature.
    top_p: float = 1.0                  # Top p sampling.
    frequency_penalty: Optional[float] = 0.0    # Frequency penalty for OpenAI.
    presence_penalty: Optional[float] = 0.0     # Presence penalty for OpenAI.
    lm_batch_size: int = 7              # Batch size for prompting the LLM.
    lm_delay: int = 15                  # Delay between prompting the LLM.


# these will be accessible directly as attributes of AllParams objects.
_short_circuited_args_: list[str] = [
    'label',
    'data_root',
    'seed',
    'gpu',
    'dataset',
    'split',
    'train_split',
    'test_split',
    'lm_name',
    'n_shots',
    'n_cands',
    'n_test',
    'selector_type',
]

@attr.s(auto_attribs=True)
class AllParams(Parameters):
    exp: ExperimentParams = ExperimentParams()
    data: DataParams = DataParams()
    llm: LLMParams = LLMParams()
    selector: CommonSelectorArgs = CommonSelectorArgs(ES.RANDOM, n_shots=10)
    logged: bool = False

    def __getattr__(self: AllParams, name: str):
        # Create short-circuited getters for attributes in _short_circuited_args_
        if name not in _short_circuited_args_:
            return super(AllParams, self).__getattribute__(name)
        sub_params = ['exp', 'data', 'llm', 'selector']
        for sp in sub_params:
            if not sp in self.__dict__: continue
            sp = self.__dict__[sp]
            if name in attr.fields_dict(sp.__class__):
                return getattr(sp, name)

    def __setattr__(self: AllParams, name: str, value):
        # Create short-circuited setters for attributes in _short_circuited_args_
        if name not in _short_circuited_args_:
            return super(AllParams, self).__setattr__(name, value)
        sub_params = ['exp', 'data', 'llm', 'selector']
        for sp in sub_params:
            if not sp in self.__dict__: continue
            sp = self.__dict__[sp]
            if name in attr.fields_dict(sp.__class__):
                return setattr(sp, name, value)

    def to_dict(self):
        """ Serialize to a nested dict """
        return dict(
            exp=self.exp.to_dict(),
            data=self.data.to_dict(),
            llm=self.llm.to_dict(),
            selector=self.selector.to_dict(),
        )

    @classmethod
    def from_dict(cls, d: dict):
        EP = converter.structure(d['exp'], ExperimentParams)
        # DP = converter.structure(d['data'], DataParams)
        DP = converter.structure(d['data'], ds2cls[d['data']['dataset']])
        LP = converter.structure(d['llm'], LLMParams)
        SP = converter.structure(d['selector'], sel2cls[d['selector']['selector_type']])
        return cls(EP, DP, LP, SP)

    @property
    def shorthand(self: AllParams):
        """utility method for easier access"""
        return self.exp, self.data, self.llm, self.selector
    
    @property
    def selector_name(self: AllParams):
        # `get_name`` will be implemented by each selector parameters class in sel2cls
        return self.selector.get_name()

    @property
    def exp_path(self: AllParams):
        """the path to the directory for the experiment for these parameters"""
        P = self
        EP, DP, LP, SP = P.shorthand

        path = Path(self.label) / DP.get_dataset_name() / DP.get_split_name()

        # PROMPT
        prompt_name = DP.get_prompt_name()
        # prompt_name = DP.get_prompt_name(default_prompt_version[self.llm.lm_name])
        if prompt_name: path /= prompt_name

        # SELECTOR
        path /= f'{self.selector.n_shots if self.selector.n_shots != -1 else "max"}_shots/{SP.selector_type}'

        if SP.selector_type in ES.RANDOM:
            path = path / f'{num2str(DP.n_cands)}_cands'

        elif SP.selector_type in [ES.COSINE, ES.BERTSCORE, ES.GIST_BERTSCORE, ES.STRUCT, ES.LF_COVERAGE]:
            selector_name = self.selector_name
            path = path / f'{num2str(DP.n_cands)}_cands-{selector_name}'

        path /= f's{self.exp.seed}'

        # LM
        # path /= self.llm.lm_name.split("/")[-1]
        path /= self.llm.lm_name.name

        return path

    @property
    def output_dir(self: AllParams) -> Path:
        return self.exp.output_root / self.exp_path

    @property
    def testname(self: AllParams) -> str:
        DP = self.data
        test_name_parts = []
        if DP.split:
            test_name_parts.append(DP.split)
        else:
            test_name_parts.append(DP.test_split)
        if DP.n_test != -1: test_name_parts.append(f'{DP.n_test}')
        return '-'.join(test_name_parts)

    @property
    def outputname(self: AllParams) -> Path:
        return self.testname

    @property
    def resultsfile(self: AllParams) -> Path:
        return self.output_dir / f'{self.outputname}.json'

    @property
    def logfile(self: AllParams) -> Path:
        return self.output_dir / f'{self.outputname}.log'

    @property
    def outfile(self: AllParams) -> Path:
        return self.output_dir / f'{self.outputname}.out'

    @property
    def promptsfile(self: AllParams) -> Path:
        return self.output_dir.parent / f'{self.outputname}-prompts.json'

    @property
    def promptslogfile(self: AllParams) -> Path:
        return self.output_dir.parent / f'{self.outputname}.log'

    @property
    def promptsoutfile(self: AllParams) -> Path:
        return self.output_dir.parent / f'{self.outputname}.out'

    @property
    def cmd(self: AllParams, only_changed=False) -> str:
        """Command to run this experiment from the shell"""
        cmd = 'python driver.py'
        cmd += f' +selector={self.selector.selector_type}'
        cmd += f' +data={self.data.dataset}'
        for k, v in sorted(self.to_flattened_dict().items(), key=lambda x: str(type(x[1]))):
            if k == 'completed': continue
            if v is None:
                cmd += f' {k}=null'
            elif isinstance(v, Enum):
                cmd += f' {k}={v.name}'
            elif isinstance(v, str):
                cmd += f' {k}="{v}"'
            else:
                cmd += f' {k}={v}'
        return cmd

    @property
    def completed(self) -> bool:
        """whether this experiment has been completed"""
        resfile = self.resultsfile if not self.exp.only_prompts else self.promptsfile
        return resfile.exists()

    def completed_after(self, timestamp: float) -> bool:
        """
        Whether this experiment has been completed after a given timestamp.
        Used in run.py:run_exps_parallel to check if the experiment finished
        as a part of a batch of experiments.
        """
        resfile = self.resultsfile if not self.exp.only_prompts else self.promptsfile
        return resfile.exists() and resfile.stat().st_mtime > timestamp

    def get_lm(self: AllParams, max_tokens) -> LLM:
        """Create the LLM based on the LLM parameters in `self.lm`."""
        LP = self.llm
        generation_kwargs = dict(
            temperature=LP.temperature, max_tokens=max_tokens, top_p=LP.top_p)
        common_kwargs = dict(
            model_name=LP.lm_name.value,
            batch_size=LP.lm_batch_size,
            verbose=self.exp.debug)
        if LP.lm_type == P.OPENAI:  # OpenAI LLMs other than Turbo
            from langchain import OpenAI
            openai_key = [l.strip() for l in open(LP.openai_keys_file).readlines()][0]
            from constants import openai_lms
            if LP.lm_name not in openai_lms:
                print(f'Using openai API for {LP.lm_name}')
                import openai
                openai.api_key = "EMPTY"
                openai.api_base = LP.lm_url
                models = openai.Model.list()
                common_kwargs['model_name'] = models['data'][0]['id']
                common_kwargs['stream'] = False
            return OpenAI(
                **common_kwargs, openai_api_key=openai_key,
                request_timeout=1000, base_delay=LP.lm_delay, keep_trying=True,
                frequency_penalty=LP.frequency_penalty,
                presence_penalty=LP.presence_penalty,
                **generation_kwargs,)
        elif LP.lm_type == P.OPENAI_CHAT:   # for turbo
            from langchain import OpenAIChat
            openai_key = [l.strip() for l in open(LP.openai_keys_file).readlines()][0]
            return OpenAIChat(
                **common_kwargs, openai_api_key=openai_key,
                request_timeout=1000, base_delay=LP.lm_delay, keep_trying=True,
                frequency_penalty=LP.frequency_penalty,
                presence_penalty=LP.presence_penalty,
                **generation_kwargs,)
        elif LP.lm_type == P.HUGGINGFACE:   # for huggingface LLMs like Neo, LLaMA, Starcoder.
            from langchain.llms.huggingface import HuggingFace
            generation_kwargs['max_new_tokens'] = generation_kwargs.pop('max_tokens')
            return HuggingFace.from_model_name(
                **common_kwargs, task='text-generation', device=self.exp.gpu, cache=False,
                generation_kwargs=generation_kwargs | dict(do_sample=LP.do_sample))
        elif LP.lm_type == P.OPT_SERVER:    # for OPT
            from langchain.llms.alpa import OptAlpaServer
            return OptAlpaServer(**common_kwargs, url=LP.lm_url, **generation_kwargs)
        else:
            raise ValueError(f'Unknown lm_type: {LP.lm_type}')

cs = ConfigStore.instance()
cs.store(name="config", node=AllParams)
for selector_type, params_cls in sel2cls.items():
    cs.store(group="selector", name=selector_type, node=params_cls)
for ds, ds_cls in ds2cls.items():
    cs.store(group="data", name=ds, node=ds_cls)

# test whether the AllParams object can be constructed from command-line.
@hydra.main(version_base=None, config_name="config")
def test(cfg: AllParams) -> None:
    P: AllParams = OmegaConf.to_object(cfg)
    print(P.selector_name)
    print(P.to_flattened_dict())
    print(P.cmd)
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    test()
