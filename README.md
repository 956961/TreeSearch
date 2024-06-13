# Official Repo of Language Agent Tree Search (LATS) - ICML 2024

<p>
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Python-3.7+-1f425f.svg?color=purple">
    </a>
    <a href="https://copyright.illinois.edu/">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
    </a>
</p>

![teaser](pics/teaser.png)

Official implementation for ICML 2024 paper [Language Agent Tree Search Unifies Reasoning Acting and Planing in Language Models](https://arxiv.org/abs/2310.04406) with code, prompts, model outputs. 

More can be found at our [project website](https://andyz245.github.io/LanguageAgentTreeSearch/) or [paper](https://arxiv.org/abs/2310.04406)

Check out our demo, CodeLATS at our [demo](https://huggingface.co/spaces/AIatUIUC/CodeLATS/tree/main)

For a more general implementation for your application, please look at the LangChain implementation in LangGraph.
https://github.com/langchain-ai/langgraph/tree/main/examples/lats 


### Reasoning + Acting (HotPotQA)

#### Setup

To get started:

1. Clone this repo
```bash
git clone https://github.com/956961/TreeSearch && cd LanguageAgentTreeSearch/hotpot
```

2. Install the module dependencies into your environment:
```bash
pip install -r requirements.txt
```

3. Set the scripts and run paper experiments
```bash
sh lats.sh
```

- ``--n_generate_sample``: number of times to prompt during expansion/sampling
- ``--n_evaluate_sample``: number of times to prompt for state evaluation
- ``--iterations``: maximum number of trajectories to sample



## Trajectories
``programming/root/`` contains all the trajectories from the paper's experiments on programming. Please use get_acc.py with the log path to get the actual accuracy. HotPotQA and WebShop logs were too large to upload, feel free to email if interested.

