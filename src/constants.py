"""Contains various Enums and constants to avoid raw strings everywhere!"""

from enum import Enum

class Task(str, Enum):
    SEMPARSE = 'Semantic Parsing'
    NLI = 'NLI'
    SENTIMENT = 'Sentiment Analysis'
    PARAPHRASE = 'Paraphrase Detection'
    COMMONSENSE = 'Commonsense Reasoning'
    COREF = 'Coreference Resolution'
    SUMMARIZATION = 'Summarization'
    CLOSEDQA = 'Closed-book QA'
    RC = 'Reading Comprehension'
    MISC = 'Misc'
    COT = 'CoT Reasoning'

class Dataset(str, Enum):
    ALPACA = 'alpaca-plus'
    FLAN = 'flan2021'
    T0 = 't0'

    # Semantic Parsing
    ATIS = 'atis'
    GEOQUERY = 'geoquery'
    OVERNIGHT = 'overnight'
    SMCALFLOW = 'smcalflow'
    SMCALFLOW_CS = 'smcalflow-cs'
    COGS = 'cogs'
    CFQ = 'cfq'
    SPIDER = 'spider'
    BREAK = 'break'
    MTOP = 'mtop'

    # NLI
    QNLI = 'qnli'
    MNLI = 'mnli'
    SNLI = 'snli'
    RTE = 'rte'
    WANLI = 'wanli'     # https://huggingface.co/datasets/alisawuffles/WANLI
    XNLI = 'xnli'       # https://huggingface.co/datasets/xnli
    MEDNLI = 'mednli'   # https://huggingface.co/datasets/medarc/mednli

    # Sentiment Analysis
    SST2 = 'sst2'
    YELP = 'yelp_polarity'
    YOUTUBE = 'youtube'
    SST5 = 'sst5'
    ROTTEN_TOMATOES = 'rotten_tomatoes'

    # Paraphrase Detection
    MRPC = 'mrpc'
    QQP = 'qqp'
    PAWS = 'paws'
    PAWSX = 'pawsx'     # https://huggingface.co/datasets/paws-x

    # Commonsense Reasoning
    CMSQA = 'cmsqa'
    COPA = 'copa'
    SWAG = 'swag'
    HELLASWAG = 'hellaswag'
    PIQA = 'piqa'

    # Summarization
    AGNEWS = 'agnews'

    # Reading Comprehension
    BOOLQ = 'boolq'
    DROP = 'drop'

    # Misc
    COLA = 'cola'
    TWEET = 'tweet_eval'

    # CoT
    GSM8K = 'gsm8k'
    
    AESLC = 'aeslc'
    DART = 'dart'
    NL2BASH = 'nl2bash'
    
D = Dataset
T = Task

category2datasets = {
    T.SEMPARSE: [D.GEOQUERY, D.SMCALFLOW_CS, D.ATIS, D.OVERNIGHT, D.BREAK, D.MTOP, D.CFQ, D.COGS, D.SPIDER],
    T.NLI: [D.QNLI, D.MNLI, D.RTE, D.WANLI, D.XNLI, D.MEDNLI],
    T.SENTIMENT: [D.SST2, D.YELP, D.SST5, D.ROTTEN_TOMATOES, D.YOUTUBE],
    T.PARAPHRASE: [D.MRPC, D.QQP, D.PAWS, D.PAWSX],
    T.COMMONSENSE: [D.CMSQA, D.COPA, D.SWAG, D.HELLASWAG, D.PIQA],
    T.SUMMARIZATION: [D.AGNEWS],
    T.COT: [D.GSM8K],
    T.RC: [D.BOOLQ, D.DROP],
    T.MISC: [D.COLA, D.TWEET],
}
heldout_datasets = [
    D.WANLI,
    D.XNLI,
    D.MEDNLI,
]

dataset2category = {d: c for c, ds in category2datasets.items() for d in ds}

class ExSel(str, Enum):
    RANDOM = 'random'
    BERTSCORE = 'bertscore'
    GIST_BERTSCORE = 'gist_bertscore'
    STRUCT = 'structural'
    COSINE = 'cosine'
    LF_COVERAGE = 'lf_coverage'
    EPR = 'epr'
    CEIL = 'ceil'
    LLMR = 'llmr'

class LMType(str, Enum):
    OPENAI = 'openai'
    OPENAI_CHAT = 'openai_chat'
    OPT_SERVER = 'opt_server'
    HUGGINGFACE = 'huggingface'

class LLM(str, Enum):
    NEO = 'EleutherAI/gpt-neo-2.7B'
    LLAMA7B = 'llama-7B'
    LLAMA13B = 'llama-13B'
    LLAMA30B = 'llama-30B'
    STARCODER = 'bigcode/starcoder'
    MISTRAL = 'mistralai/Mistral-7B-v0.1'
    ZEPHYR = 'HuggingFaceH4/zephyr-7b-alpha'

    BABBAGE_002 = 'babbage-002'
    DAVINCI_002 = 'davinci-002'
    CODE_CUSHMAN_001 = 'code-cushman-001'
    CODE_DAVINCI_002 = 'code-davinci-002'
    TEXT_DAVINCI_002 = 'text-davinci-002'
    TEXT_DAVINCI_003 = 'text-davinci-003'
    TURBO = 'gpt-3.5-turbo'
    TURBO_JUNE = 'gpt-3.5-turbo-0613'
    GPT4 = 'gpt-4-0314'
    MAJORITY = 'majority'
    
    DOLLY3B = 'databricks/dolly-v2-3b'
    DOLLY7B = 'databricks/dolly-v2-7b'

openai_lms = [LLM.BABBAGE_002, LLM.DAVINCI_002, LLM.CODE_CUSHMAN_001, LLM.CODE_DAVINCI_002, LLM.TEXT_DAVINCI_002, LLM.TEXT_DAVINCI_003, LLM.TURBO, LLM.TURBO_JUNE, LLM.GPT4]
chat_lms = [LLM.TURBO, LLM.TURBO_JUNE]

context_length_limit = {
    LLM.CODE_CUSHMAN_001: 2048,
    LLM.CODE_DAVINCI_002: 8001,
    LLM.TEXT_DAVINCI_002: 4096,
    LLM.TEXT_DAVINCI_003: 4096,
    LLM.TURBO: 16385,
    LLM.TURBO_JUNE: 4000,
    LLM.GPT4: 8192,

    LLM.BABBAGE_002: 16384,
    LLM.DAVINCI_002: 16384,
    LLM.NEO: 2048,
    LLM.LLAMA7B: 2048,
    LLM.LLAMA13B: 2048,
    LLM.STARCODER: 7000,
    LLM.MISTRAL: 8192,
    LLM.ZEPHYR: 8192,
    LLM.MAJORITY: 100000,
    
    LLM.DOLLY3B: 2048,
    LLM.DOLLY7B: 2048,
}

mwp_datasets = [D.GSM8K]
