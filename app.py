import os

import streamlit as st
from dataclasses import dataclass, asdict
from ctransformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import hf_hub_download

from utils import parse_json_config


config = parse_json_config('config.json')
model_filepath = f'{config["models_directory"]}/{config["model_filename"]}'
print(config)
if not os.path.isfile(model_filepath):
    print(f'Model not found ({model_filepath}), downloading the model.')
    hf_hub_download(
        repo_id=config['repo_id'],
        filename=config['model_filename'],
        local_dir=config['models_path'],
    )

@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    max_new_tokens: int
    seed: int
    reset: bool
    stream: bool
    threads: int
    stop: list[str]


def format_prompt(system_prompt, user_prompt):
    """format prompt based on: https://huggingface.co/spaces/mosaicml/mpt-30b-chat/blob/main/app.py"""

    system_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    user_prompt = f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
    assistant_prompt = f"<|im_start|>assistant\n"

    return f"{system_prompt}{user_prompt}{assistant_prompt}"


def generate(
        llm: AutoModelForCausalLM,
        generation_config: GenerationConfig,
        system_prompt,
        user_prompt,
):
    """run model inference, will return a Generator if streaming is true"""

    return llm(
        format_prompt(
            system_prompt,
            user_prompt,
        ),
        **asdict(generation_config),
    )

st.title(f'{config["model_name"]}')
prompt = st.text_input('Your prompt goes here')

config = AutoConfig.from_pretrained(config['model_name'], context_length=600)
llm = AutoModelForCausalLM.from_pretrained(
    model_filepath,
    model_type='mpt',
    config=config,
)

system_prompt = "A conversation between a user and an LLM-based AI assistant named Local Assistant. Local Assistant gives helpful and honest answers."

generation_config = GenerationConfig(
    temperature=.2,
    top_k=0,
    top_p=.9,
    repetition_penalty=1.0,
    max_new_tokens=512,  # adjest as needed
    seed=42,
    reset=False,
    stream=True,  # reset history (cache)
    # threads=int(os.cpu_count() / 2),  # adjust for your CPU
    threads=os.cpu_count() - 2,  # adjust for your CPU
    stop=['<|im_end|>', '|<'],
)

user_prefix = '[user]: '
assistant_prefix = f'[assistant]: '


if prompt:
    ans = generate(llm, generation_config, system_prompt, prompt.strip())
    ans = ' '.join(ans)
    st.write(ans)
