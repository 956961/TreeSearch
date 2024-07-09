import json
from typing import Dict, Any
from langchain import HuggingFacePipeline
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class Llama2JsonAgent:
    def __init__(self, model_name: str):
        # 初始化Llama 2模型
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 创建文本生成管道
        pipe = pipeline(
            "text-generation",
            model=model, 
            tokenizer=tokenizer, 
            max_length=4000,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # 创建LangChain的HuggingFacePipeline
        self.llm = HuggingFacePipeline(pipeline=pipe)
        self.json_data: Dict[str, Any] = {}

    def load_json(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.json_data = json.load(file)
            return "JSON文件已成功加载。"
        except Exception as e:
            return f"加载JSON文件时出错: {str(e)}"

    def get_json_keys(self):
        return list(self.json_data.keys())

    def get_json_value(self, key: str):
        return json.dumps(self.json_data.get(key, "Key not found"), ensure_ascii=False)

    def setup_agent(self):
        tools = [
            Tool(
                name="GetJSONKeys",
                func=self.get_json_keys,
                description="当你需要知道JSON数据中有哪些键时使用这个工具"
            ),
            Tool(
                name="GetJSONValue",
                func=self.get_json_value,
                description="当你需要获取JSON数据中某个键的值时使用这个工具。输入应该是键的名称。"
            )
        ]

        self.agent = initialize_agent(
            tools, 
            self.llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            verbose=True
        )

    def query(self, user_query: str) -> str:
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            你是一个JSON数据分析助手。请回答用户关于JSON数据的问题。
            如果需要，你可以使用提供的工具来获取信息。
            
            用户查询: {query}
            
            请提供准确、简洁的回答:
            """
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return self.agent.run(chain.run(query=user_query))

def main():
    # 初始化agent
    agent = Llama2JsonAgent("meta-llama/Llama-2-7b-hf")

    # 加载JSON文件
    print(agent.load_json("../Meta-Llama-3-8B/sample.json"))

    # 设置agent
    agent.setup_agent()

    while True:
        user_input = input("请输入您的查询 (输入'退出'结束): ")
        if user_input.lower() == '退出':
            break
        
        response = agent.query(user_input)
        print("Agent回答:", response)
        print()

if __name__ == "__main__":
    main()