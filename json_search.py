import json
import torch
from langchain.tools import BaseTool
from typing import List, Dict, Any, Type
from pydantic import BaseModel, Field
import os
from typing import Any
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class JSONToolInput(BaseModel):
    query: str = Field(..., description="The query to run on the JSON data")

class JSONTool(BaseTool):
    name: str = "JSON_tool"
    description: str = "A tool for performing operations on JSON data. Available operations: find, count, filter, sort, aggregate"
    args_schema: Type[BaseModel] = JSONToolInput
    json_data: List[Dict[str, Any]] = Field(default_factory=list, description="The JSON data to operate on")

    def _run(self, query: str) -> str:
        query_lower = query.lower()
        
        if "find" in query_lower:
            return self._find(query)
        elif "count" in query_lower:
            return self._count(query)
        elif "filter" in query_lower:
            return self._filter(query)
        elif "sort" in query_lower:
            return self._sort(query)
        elif "aggregate" in query_lower:
            return self._aggregate(query)
        else:
            return json.dumps(self.json_data, indent=2)

    def _find(self, query: str) -> str:
        # Example: "find items where name = John"
        parts = query.split()
        key = parts[parts.index("where") + 1]
        value = " ".join(parts[parts.index("=") + 1:])
        results = [item for item in self.json_data if str(item.get(key)) == value]
        return json.dumps(results, indent=2)

    def _count(self, query: str) -> str:
        # Example: "count items" or "count items where age > 30"
        if "where" in query:
            parts = query.split()
            key = parts[parts.index("where") + 1]
            op = parts[parts.index("where") + 2]
            value = parts[parts.index("where") + 3]
            if op == '>':
                results = [item for item in self.json_data if item.get(key, 0) > float(value)]
            elif op == '<':
                results = [item for item in self.json_data if item.get(key, 0) < float(value)]
            elif op == '=':
                results = [item for item in self.json_data if str(item.get(key)) == value]
            return str(len(results))
        else:
            return str(len(self.json_data))

    def _filter(self, query: str) -> str:
        # Example: "filter items where age > 30"
        parts = query.split()
        key = parts[parts.index("where") + 1]
        op = parts[parts.index("where") + 2]
        value = parts[parts.index("where") + 3]
        if op == '>':
            results = [item for item in self.json_data if item.get(key, 0) > float(value)]
        elif op == '<':
            results = [item for item in self.json_data if item.get(key, 0) < float(value)]
        elif op == '=':
            results = [item for item in self.json_data if str(item.get(key)) == value]
        return json.dumps(results, indent=2)

    def _sort(self, query: str) -> str:
        # Example: "sort items by age descending"
        parts = query.split()
        key = parts[parts.index("by") + 1]
        order = parts[-1] if parts[-1] in ["ascending", "descending"] else "ascending"
        sorted_data = sorted(self.json_data, key=lambda x: x.get(key, 0), reverse=(order == "descending"))
        return json.dumps(sorted_data, indent=2)

    def _aggregate(self, query: str) -> str:
        # Example: "aggregate sum of age"
        parts = query.split()
        operation = parts[1]
        key = parts[3]
        if operation == "sum":
            result = sum(item.get(key, 0) for item in self.json_data)
        elif operation == "average":
            values = [item.get(key, 0) for item in self.json_data]
            result = sum(values) / len(values) if values else 0
        elif operation == "max":
            result = max(item.get(key, 0) for item in self.json_data)
        elif operation == "min":
            result = min(item.get(key, 0) for item in self.json_data)
        return str(result)

    def _arun(self, query: str):
        raise NotImplementedError("JSONTool does not support async")

# 使用示例
from langchain import HuggingFacePipeline
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

# 加载你的JSON数据
with open('../Meta-Llama-3-8B/sample.json', 'r') as f:
    json_data = json.load(f)

# 创建JSON工具
json_tool = JSONTool(json_data=json_data)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel
# 创建语言模型
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM

class FakeLLM(LLM):
    """Fake LLM wrapper for testing."""

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """First try to lookup in query map, else return simple response."""
        return f"This is a simulated response to: {prompt}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"
access_token = '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'
# 使用FakeLLM替换OpenAI
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
def get_llama_llm(model_name: str, max_token: int =40960, temperature: float=0.1) -> BaseLanguageModel:
    model_name = model_name.capitalize()
    if model_name not in LLAMA_HF:
        raise ValueError(f'model_name should be one of {LLAMA_HF}, not {model_name}')
    model_name = f'meta-llama/{model_name}'

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            token=access_token
                                            )
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map='cuda',
                                                torch_dtype=torch.float16,
                                                 max_position_embeddings = 8096,
                                                token=access_token
                                                )
    print(max_token)
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

llm = get_llama_llm('Llama-2-7b-hf')

agent = initialize_agent(
    [Tool(name="JSON_tool", func=json_tool.run, description=json_tool.description)],
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

# 使用agent
while True:
    query = input("Ask a question about your JSON data (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = agent.run(query)
    print(f"Answer: {response}\n")


