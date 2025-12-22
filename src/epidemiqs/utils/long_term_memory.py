from typing import List, Optional
from os.path import join as ospj
from pydantic_core import to_jsonable_python
from pydantic_ai import ModelMessagesTypeAdapter, Agent
import json
from json import JSONDecodeError
from dotenv import load_dotenv
from epidemiqs.utils.config import get_settings
load_dotenv()


class LTM: # Long Term Memory
    def __init__(self, agent_name:str, path:str=ospj("output","LTM.json")):
        self.agent_name=agent_name
        self.path=path
        self.memory=self.load_LTM()

    def save_LTM(self, agent_memory: List=[], agent_name: Optional[str]=None, path: Optional[str] = None):
        file_path = path or self.path
        agent_key = agent_name or self.agent_name
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}

        existing_data[agent_key] = to_jsonable_python(agent_memory)
        self.memory=agent_memory

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)
    
    @classmethod
    def save_LTM_direct(cls, agent_name: str, agent_memory: List, path: str = ospj("output", "LTM.json")):
        """save agent memory without creating an instance."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, JSONDecodeError):
            existing_data = {}

        existing_data[agent_name] = to_jsonable_python(agent_memory)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)
            
    def load_LTM(self,agent_name:Optional[str]=None, path:Optional[str]=None):
        try:
            with open(path or self.path, 'r',encoding='utf-8') as f:
                ltm_dict = json.load(f)
                agent_memory= ltm_dict.get(agent_name or self.agent_name, [])
                #print(f"Loaded LTM for agent {agent_name or self.agent_name}: {agent_memory}")
            return ModelMessagesTypeAdapter.validate_python(agent_memory)
        
        except FileNotFoundError:
            return None 

if __name__ == "__main__":
    agent=Agent('openai:gpt-4.1-nano')

    #ltm=LTM(agent_name='test_agent4')
    print(LTM(agent_name='test_agent4').memory)
    """
    #print(ltm.memory)
    #agent_memory=None
    for i in range(3):
        result=agent.run_sync(f"what have you done so far?",message_history=ltm.memory)
        print(result.output)
        ltm.save_LTM(agent_memory=result.all_messages())    
        agent_memory=ltm.load_LTM()
    #print(agent_memory)"""