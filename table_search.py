DF_PROMPT2 = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should make a valid python command as input. You must use print the output using the `print` function at the end.
You should use the `to_markdown` function when you print a pandas object.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Input: the valid python code only using the Pandas library
Observation: the result of python code
... (this Thought/Input/Observation can repeat N times)
Final Thought: you should think about how to answer the question based on your observation
Final Answer: the final answer to the original input question. If you can't answer the question, say `nothing`

The index of the dataframe must be be one of {df_index}. If it's not in the index you want, skip straight to Final Thought.
{information}

Begin!

Question: What is the head of df? If you extracted successfully, derive 'success' as the final answer
Thought: To get the head of a DataFrame, we can use the pandas function head(), which will return the first N rows. By default, it returns the first 5 rows.
Input: 
``` 
import pandas as pd
import json
print(df.head().to_markdown())
```
Observation: {df_head}
Final Thought: The head() function in pandas provides the first 5 rows of the DataFrame.
Final Answer: success

Question: {question}
{agent_scratchpad}
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline
from langchain.base_language import BaseLanguageModel

access_token = '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'
LLAMA_HF = [
    'Llama-2-7b',
    'Llama-2-7b-hf',
    'Llama-2-7b-chat',
    'Llama-2-7b-chat-hf',
    'Llama-2-13b',
    'Llama-2-13b-hf',
    'Llama-2-13b-chat',
    'Llama-2-13b-chat-hf',
    'Llama-2-70b',
    'Llama-2-70b-hf',
    'Llama-2-70b-chat',
    'Llama-2-70b-chat-hf',
]


def get_llama_llm(model_name: str, max_token: int =4096, temperature: float=0.1) -> BaseLanguageModel:
    model_name = model_name.capitalize()
    if model_name not in LLAMA_HF:
        raise ValueError(f'model_name should be one of {LLAMA_HF}, not {model_name}')
    model_name = f'meta-llama/{model_name}'

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            token=access_token
                                            )
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map='auto',
                                                torch_dtype=torch.float16,
                                                token=access_token
                                                )

    pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                do_sample=True,
                top_k=30,
                max_new_tokens=max_token,
                eos_token_id=tokenizer.eos_token_id
                )

    llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':temperature})
    return llm

config2 = {'search_internet': True,
 'verbose': True,
 'handle_errors': True,
 'temperature': 0.0,
 'model': 'gpt-4',
 'model_dir': '/home/kemove/anaconda3/envs/py39_deco/lib/python3.9/site-packages/chatmof/database/load_model/',
 'structure_dir': '/home/kemove/anaconda3/envs/py39_deco/lib/python3.9/site-packages/chatmof/database/structures/',
 'data_dir': '/home/kemove/anaconda3/envs/py39_deco/lib/python3.9/site-packages/chatmof/database/structures/coremof/',
 'hmof_dir': '/home/kemove/anaconda3/envs/py39_deco/lib/python3.9/site-packages/chatmof/database/structures/hMOF/',
 'generate_dir': '/home/kemove/anaconda3/envs/py39_deco/lib/python3.9/site-packages/chatmof/database/structures/generate',
 'lookup_dir': '/home/kemove/anaconda3/envs/py39_deco/lib/python3.9/site-packages/chatmof/database/tables/coremof.xlsx',
 'max_iteration': 3,
 'token_limit': False,
 'buildingblock_dir': '/home/kemove/anaconda3/envs/py39_deco/lib/python3.9/site-packages/chatmof/database/tables/mofkey.xlsx',
 'max_length_in_predictor': 30,
 'accelerator': 'cuda',
 'num_genetic_cycle': 3,
 'num_parents': 200,
 'logger': 'generate_mof.log',
 'topologies': ['pcu', 'dia', 'acs', 'rtl', 'cds', 'srs', 'ths', 'bcu', 'fsc']}

import os
import re
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional, Callable

from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import tiktoken



class TableSearcher2(Chain):
    """Tools that search csv using Pandas agent"""
    llm_chain: LLMChain
    df: pd.DataFrame
    encode_function: Callable
    num_max_data: int = 200
    input_key: str = 'question'
    output_key: str = 'answer'

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _parse_output(self, text:str) -> Dict[str, Any]:
        thought = re.search(r"(?<!Final )Thought:\s*(.+?)\s*(Observation|Final|Input|Thought|Question|$)", text, re.DOTALL)
        input_ = re.search(r"Input:\s*(?:```|`)?(.+?)(?:```|`)?\s*(Observation|Final|Input|Thought|Question|$)", text, re.DOTALL)
        final_thought = re.search(r"Final Thought:\s*(.+?)\s*(Observation|Final|Input|Thought|Question|$)", text, re.DOTALL)
        final_answer = re.search(r"Final Answer:\s*(.+?)\s*(Observation|Final|Input|Thought|Question|$)", text, re.DOTALL)
        observation = re.search(r"Observation:\s*(.+?)\s*(Observation|Final|Input|Thought|Question|$)", text, re.DOTALL)
        
        if (not input_) and (not final_answer):
            raise ValueError(f'unknown format from LLM: {text}')
        
        return {
            'Thought': (thought.group(1) if thought else None),
            'Input': (input_.group(1).strip() if input_ else None),
            'Final Thought' : (final_thought.group(1) if final_thought else None),
            'Final Answer': (final_answer.group(1) if final_answer else None),
            'Observation': (observation.group(1) if observation else None),
        }
    
    def _clear_name(self, text:str) -> str:
        remove_list = ['_clean_h', '_clean', '_charged', '_manual', '_ion_b', '_auto', '_SL', ]
        str_remove_list = r"|".join(remove_list)
        return re.sub(rf"({str_remove_list})", "", text)
    
    @staticmethod
    def _get_df(file_path: str):
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path)
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f'table must be .csv, .xlsx, or .json, not {file_path.suffix}')

        return df
    
    def _write_log(self, action, text, run_manager):
        run_manager.on_text(f"\n[Table Searcher] {action}: ", verbose=self.verbose)
        run_manager.on_text(text, verbose=self.verbose, color='yellow')
    
    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None
    ):
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        
        return_observation = inputs.get('return_observation', False)
        information = inputs.get('information', "If unit exists, you must include it in the final output. The name of the material exists in the column \"name\".")

        agent_scratchpad = ''
        max_iteration = config2['max_iteration']

        input_ = self._clear_name(inputs[self.input_key])

        for i in range(max_iteration + 1):
            llm_output = self.llm_chain.run(
                df_index = str(list(self.df)),
                information = information,
                df_head = self.df.head().to_markdown(),
                question=input_,
                agent_scratchpad = agent_scratchpad,
                callbacks=callbacks,
                stop=['Observation:', 'Question:',]
            )

            if not llm_output.strip():
                agent_scratchpad += 'Thought: '
                llm_output = self.llm_chain.run(
                    df_index = str(list(self.df)),
                    information = information,
                    df_head = self.df.head().to_markdown(),
                    question=input_,
                    agent_scratchpad = agent_scratchpad,
                    callbacks=callbacks,
                    stop=['Observation:', 'Question:',]
                )
            
            #if llm_output.endswith('Final Answer: success'):
            if re.search(r'Final Answer: (success|.* above|.* success|.* succeed|.* DataFrames?).?$', llm_output):
                thought = f'Final Thought: we have to answer the question `{input_}` using observation\n'
                agent_scratchpad += thought
                llm_output = self.llm_chain.run(
                    df_index = str(list(self.df)),
                    df_head = self.df.head().to_markdown(),
                    question=input_,
                    information = information,
                    agent_scratchpad = agent_scratchpad,
                    callbacks=callbacks,
                    stop=['Observation:', 'Question:',]
                )
                llm_output = thought + llm_output

            output = self._parse_output(llm_output)

            if output['Final Answer']:
                if output['Observation']:
                    raise ValueError(llm_output)
                
                self._write_log('Final Thought', output['Final Thought'], run_manager)

                final_answer: str = output['Final Answer']
                #if final_answer == 'success':
                #    agent_scratchpad += 'Thought: {}\nInput: {} \nObservation: {}\n'\
                #            .format(output['Thought'], output['Input'], observation)

                #check_sentence = ' Check to see if this answer can be your final answer, and if so, you should submit your final answer. You should not do any additional verification on the answer.'
                check_sentence = ''
                if re.search(r'nothing', final_answer):
                    final_answer = 'There are no data in database.' # please use tool `predictor` to get answer.'
                elif final_answer.endswith('.'):    
                    final_answer += check_sentence
                else:
                    final_answer = f'The answer for question "{input_}" is {final_answer}.{check_sentence}'

                self._write_log('Final Answer', final_answer, run_manager)
                return {self.output_key: final_answer}
            
            elif i >= max_iteration:
                final_answer = 'There are no data in database'
                self._write_log('Final Thought',
                                output['Final Thought'], run_manager)
                self._write_log('Final Answer', final_answer, run_manager)

                return {self.output_key: final_answer}

            else:
                self._write_log('Thought', output['Thought'], run_manager)
                self._write_log('Input', output['Input'], run_manager)
                
            pytool = PythonAstREPLTool(locals={'df':self.df})
            observation = str(pytool.run(output['Input'])).strip()

            num_tokens = self.encode_function(observation)
            
            if return_observation:
                if "\n" in observation:
                    self._write_log('Observation', "\n"+observation, run_manager)
                else:
                    self._write_log('Observation', observation, run_manager)
                return {self.output_key: observation}
            
            #if num_tokens > 3400:
                #raise ValueError('The number of tokens has been exceeded.')
                # observation = f"The number of tokens has been exceeded. To reduce the length of the message, please modify code to only pull up to {self.num_max_data} data."
                # self.num_max_data = self.num_max_data // 2

            if "\n" in observation:
                self._write_log('Observation', "\n"+observation, run_manager)
            else:
                self._write_log('Observation', observation, run_manager)

            agent_scratchpad += 'Thought: {}\nInput: {} \nObservation: {}'\
                .format(output['Thought'], output['Input'], observation)

        raise AssertionError('Code Error! please report to author!')

    @classmethod
    def from_filepath(
        cls,
        llm: BaseLanguageModel,
        file_path: Path = Path(config2['lookup_dir']),
        prompt: str = DF_PROMPT2,
        **kwargs
    ) -> Chain:
        template = PromptTemplate(
            template=prompt,
            input_variables=['df_index', 'information', 'df_head', 'question', 'agent_scratchpad']
        )
        llm_chain = LLMChain(llm=llm, prompt=template)
        df = cls._get_df(file_path)
        encode_function = llm.get_num_tokens
        return cls(llm_chain=llm_chain, df=df, encode_function=encode_function, **kwargs)
    
    @classmethod
    def from_dataframe(
        cls,
        llm: BaseLanguageModel,
        dataframe: pd.DataFrame,
        prompt: str = DF_PROMPT2,
        **kwargs
    ) -> Chain:
        template = PromptTemplate(
            template=prompt,
            input_variables=['df_index', 'information', 'df_head', 'question', 'agent_scratchpad']
        )
        llm_chain = LLMChain(llm=llm, prompt=template)

        encode_function = llm.get_num_tokens
        return cls(llm_chain=llm_chain, df=dataframe, encode_function=encode_function, **kwargs)


import os
from typing import Any
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool



def _get_search_csv(
        llm: BaseLanguageModel,
        file_path: str = config2['lookup_dir'],
        verbose: bool = False,
        **kwargs: Any) -> BaseTool:

    return Tool(
        name="search_csv",
        description=(
                "A tools that extract accurate properties from a look-up table in the database. "
                #"input must be provided in the form of a full sentence. "
                "The input must be a detailed full sentence to answer the question."
        ),
        func=TableSearcher2.from_filepath(
            llm=llm, file_path=file_path, verbose=verbose
        ).run
    )


_get_search_csv(llm)