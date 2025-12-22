import os
import time
import json
import uuid
from datetime import datetime, timezone
from os.path import join as ospj
from epidemiqs.utils.utils import log_tokens
from dataclasses import dataclass, field
from pydantic_ai import Agent, RunContext,ModelRetry
from typing import List, Dict, Optional, Union, Literal
from epidemiqs.utils.config import get_settings,Settings
from epidemiqs.utils.llm_models import choose_model
from termcolor import colored, cprint
from epidemiqs.agents.task_expert_agents import OnlineRetrieverExpert,LiteratureReviewExpert,MathExpert
import traceback
from pydantic_ai.tools import Tool, ToolDefinition
from epidemiqs.utils.long_term_memory import LTM
from epidemiqs.utils.llm_models import choose_model
from pydantic import BaseModel, Field, ConfigDict, model_validator
from pydantic.dataclasses import dataclass
from epidemiqs.agents.prompts import system_prompts
from epidemiqs.agents.output_structures import EpidemicInfo, DiscoveryScientistInfo, ReflectResult
import asyncio
print_lock = asyncio.Lock()




def validate_result(ctx: RunContext[None], result: EpidemicInfo) -> EpidemicInfo:
    if result.network_path:  
        if not os.path.exists(result.network_path):
            raise ModelRetry(
                f"Network path {result.network_path} does not exist. Please provide a valid path."
            )

    return result



class DiscoveryScientist():
    def __init__(self,llm=None,
                cfg:Settings=None,
                 phase:str="Discovery",
                 name:str="DiscoveryScientist",
                 output_type:EpidemicInfo=EpidemicInfo,
                 scinetist_output_type:DiscoveryScientistInfo=DiscoveryScientistInfo,
                 system_prompt:str=system_prompts["discovery_scientist"],
                 repo:str=ospj(os.getcwd(),"output"),
                 ltm_path:str=None
                 ):
        self.cfg=cfg if cfg is not None else get_settings()
        self.llm=llm if llm is not None else choose_model(self.cfg)["scientists"]
        self.name= name 
        self.memory = LTM(agent_name=self.react_module.name, path=ltm_path).memory if ltm_path is not None else None
        self.ltm_path=ltm_path or ospj("output","LTM.json")
        self.conv_length = [] 
        self.input_tokens=0
        self.output_tokens=0
        tools = [
        Tool(self.search_web_online, takes_ctx=False),
        Tool(self.literature_review, takes_ctx=False),
        Tool(self.ask_mathematician_agent, takes_ctx=False),
        ]

        if self.cfg.workflow.copilot:
            tools.append(Tool(self.ask_user, takes_ctx=False))
        self.react_module = Agent(
            model=self.llm,
            name="DiscoveryScientist",
            system_prompt=system_prompt if system_prompt is not None else system_prompts["discovery_scientist"],
            output_type=scinetist_output_type,
            end_strategy="exhaustive",
            
            model_settings={
            #    
                "parallel_tool_calls ": False,
                        },
            output_retries=cfg.workflow.no_retries,
            retries=50,
        tools=tools
        ,
        )
        self.reflect_module = Agent(name="ReflectModule",
            model=self.llm,
            system_prompt=system_prompt+"You must focus exclusively on reflecting to ensure the accuracy,completeness and no devation of the result generated, if you are satisfied, set the revision_needed to False, and if revision is needed  provide step-by-step instructions to correct any mistakes or inaccuracies in the response.",
            output_type=ReflectResult,
        )
        self.plan_module = Agent(
            name="ThinkModule",
            model=self.llm,
            system_prompt=system_prompt+"you must Focus exclusively on planning and reasoning about the task at hand to generate a step-by-step plan to gather of how the information needed to complete the task.",
            
        )
        self.output_type = output_type
        self.math_agent = MathExpert()
        self.math_history = None
        self.mathematical_solution = ""
        self.tokens=0
        self.phase = phase
        self.repo=repo
        self.literature_reviewer=LiteratureReviewExpert()
        self.online_retriever=OnlineRetrieverExpert()
        self.user_query=""
        self.no_tool_exec_iters=0
        self.no_reflect_iters=0
        if not os.path.exists(self.repo):
            os.makedirs(self.repo)
        print(colored(f"""Discovery Team initialized with agents:\n{self.react_module.name} powered by: {self.react_module.model.model_name},\n{self.math_agent.react_module.name} powered by:{self.math_agent.react_module.model.model_name},\n{self.literature_reviewer.react_module.name} powered by:{self.literature_reviewer.react_module.model.model_name},\n{self.online_retriever.react_module.name} powered by:{self.online_retriever.react_module.model.model_name}
                      """, "light_blue"))
    async def forward(self, query: str) -> EpidemicInfo: #,  reflection_steps:int=1,copilot:bool=False) -> EpidemicInfo:
        self.user_query=query
        st= time.time()
        prompt=f"Now you based n the user query as:\n {query}\n,You should gather relevant infromation through you tools and your knowledge to generate required information\n"
        if self.cfg.workflow.scientist_modules.plan:
            try:
                think_prompt =prompt+ f"Now I want you to think about the user query through chain of thought, and plan how to gather the relevant information about the epidemic situation, and provide a step by step plan to gather the information. \n"
                plan = await self.plan_module.run(think_prompt, message_history=self.memory)
                self.log_agent_event("plan", think_prompt, plan.output)
                self._count_tokens(plan)
                prompt += f"sugested plan\n: {plan.output}\n"
                print(colored(f"Plan:\n {plan.output}\n","blue"))
            except Exception as e:
                error_trace = traceback.format_exc() 
                self.log_agent_event("plan", think_prompt, None, error=e)
                print(colored(f"Error in DiscoveryScientist: {str(e)}\n{error_trace}",'red'))
    
        try:
            
            result = await self.run_react(prompt)
            print(colored(f"DiscoveryScientist result:\n {result}\n","green"))

            for _ in range(self.cfg.workflow.reflection_max_iters):

                    if  self.cfg.workflow.scientist_modules.reflect:  
                        self.no_reflect_iters += 1
                        ref_prompt = f"""Now I want you to reflect on the generated response and check if  it was accurate, include all user requests and ensure there was no deviation from the user request, and all necesssary information are gathered and included.
                             If there are mistakes, inaccuracies or any tasks missed, please reason, plan to provide accurate steo by step instructions in your reflection to correct the answer. If no deviation is found, then no revision is need , revised_needed=False\n
                             user Initial Query: {query}\n
                             Generated Response:\n {result}\n"""
                        reflection = await self.reflect_module.run(ref_prompt, message_history=self.memory)
                        self.log_agent_event("reflect", ref_prompt, reflection.output.to_dict())
                        self._count_tokens(reflection)
                        
                        print(colored(f"Reflection: {reflection.output}","yellow"))
                        if reflection.output.revise_needed:
                            prompt = f"Please consider the following comments\n : {reflection}\n and provide a more accurate and precise answer according to the reflection provided..\n"
                            result = await self.run_react(prompt)
                            if self.memory and len(self.memory) > 0:
                                self.conv_length.append(len(self.memory))
                            self._trim_context_memory()
                            print(colored(f"DiscoveryScientist  Revised Result:\n {result}\n","green"))  
                        elif not reflection.output.revise_needed:
                            print(colored("No revisions needed based on reflection.", "green"))
                            break  # exit the reflection loop if no revision is needed                        
                            
                
            if self.cfg.workflow.copilot:
                user_input="no"
                while user_input.lower() not in ["yes",'y']:
                    input_text = colored(f"Based on the provided information, the generated results is as:\n {result.output}\nIf you have any comment or suggestion, please provide it here. If the output is good as it is enter \n\"yes\"\nOW, please enter your concerns:\n","white")
                    user_input=input(input_text)
                    if user_input.lower() != "yes":
                        prompt = f"Please consider the following user comments:\n {user_input}\n and provide a more accurate and precise answer according to the comments.\n"
                        result = await self.run_react(prompt)
                        print(colored(f"DiscoveryScientist  Revised Result:\n {result}\n","green"))
                        self.memory = result.all_messages()
                        
            self.tokens += self.tokens+self.literature_reviewer.tokens+self.online_retriever.tokens+self.math_agent.tokens
            log_tokens(repo=self.repo, agent_name=self.react_module.name, llm_model=self.react_module.model.model_name,input_tokens=self.input_tokens, output_tokens=self.output_tokens, total_tokens=self.input_tokens+self.output_tokens,time=time.time()-st)
            log_tokens(repo=self.repo, csv_name="tokens_by_phase.csv",agent_name=self.phase, llm_model=self.react_module.model.model_name,input_tokens=self.input_tokens, output_tokens=self.output_tokens, total_tokens=self.tokens,time=time.time()-st)
            LTM.save_LTM_direct(agent_name=self.name, agent_memory=self.memory, path=self.ltm_path)
            return self.output_type(**result.to_dict(), mathematic_full_response=self.mathematical_solution)
        except Exception as e:
            error_trace = traceback.format_exc() 
            print(colored(f"Error in DiscoveryScientist: {str(e)}\n{error_trace}",'red'))
            return f"Error in DiscoveryScientist: {str(e)}\n{error_trace}"
                
    def log_agent_event(self,
        module_name: str,
        input_data=None,
        output_data=None,
        error=None,
        meta: dict | None = None,
        log_path: str =None) -> Dict:
        """
        Append a log entry for an agent module (plan/react/reflect) to a JSONL file.
        Creates the file if it does not exist.
        
        Parameters
        ----------
        module_name : str
            "plan", "react", or "reflect"
        input_data : Any
            Input given to the module
        output_data : Any
            Output returned by the module
        error : Exception | str | None
            Error message or exception (if occurred)
        meta : dict | None
            Additional metadata (iteration id, agent name, config, etc.)
        log_path : str
            File path for JSONL log
        """
        
        log_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "module": module_name,
            "input": input_data,
            "output": output_data,
            "error": str(error) if error else None,
            "no_exec_iter": self.no_tool_exec_iters,   
            "no_reflections":self.no_reflect_iters, 
            "meta": meta or {}
        }

        # append as a JSON line
        log_path=log_path if log_path is not None else ospj(self.repo, f"{self.name}.jsonl")
        with open(ospj(self.repo, log_path), "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False))
            f.write("\n")

        return log_entry    
    
    async def run_react(self,prompt: str,max_retries: int=3) -> DiscoveryScientistInfo:
        for attempt in range(max_retries):
            try:
                result = await self.react_module.run(prompt, message_history=self.memory)
                self.memory = result.all_messages()
                self.log_agent_event("react", prompt, result.output.to_dict())
                print(colored(f"{self.name} result:\n{result.output}\n", "green"))
                self._count_tokens(result)
                return result.output  # exit the function with the result

            except Exception as e:
                if attempt == max_retries - 1:
                    message=DiscoveryScientistInfo(description=f"Discovery failed after maximum retries {max_retries} due to error: {e}.Proceeding without these information and use only user query:[{self.user_query}]",
                                                  task="discovery_failed",
                                                  goal="N/A",
                                                  pathogen="N/A",
                                                  compartment_model="N/A",
                                                  disease_type="N/A",
                                                  current_condition="N/A",
                                                  R_0=None,
                                                  network_path=None,
                                                  contact_network_structure=None,
                                                  math_solution=None,
                                                  data_path=None,
                                                  reasoning_info=f"Discovery failed after maximum retries {max_retries} due to error: {e}.Proceeding without these information and use only user query:{self.user_query}"
                                                  )
                    self.log_agent_event("react", prompt, message.to_dict(), error=e)
                    
                    return message        
    async def ask_mathematician_agent(self, question: str) -> str:
        """Ask smart mathematician when questions need mathematical analysis. Ask Only if Analytical solution is needed. the more precise and clear the question, the better the answer."""
        print(colored(f"\nAsking Mathematician: {question}...\n","magenta"))
        self.no_tool_exec_iters+=1
        try:
            answer = await self.math_agent.ask_mathematician_agent(question)
            self.mathematical_solution+=answer
            print(colored(f"Mathematician answer:\n{self.mathematical_solution}\n","yellow"))
            return answer
        except Exception as e:
            error_trace = traceback.format_exc() 
            print(colored(f"Error in ask_mathematician_agent: {str(e)}\n{error_trace}",'red'))
            return f"Math Expert is not available right now  {str(e)}\n{error_trace}"
        
        


    async def search_web_online(self, query: str) -> str:
        """     
        Ask the Online Retriever Agent to answer your query by searching online. if you need up-to-date information, just mention up-to-date in the query. 

        """
        self.no_tool_exec_iters+=1
        print(colored( f"\nSearching Web Online...\nQuery: {query} ",'blue'))
        try:
            result = await self.online_retriever.run(query)
            return result
        except Exception as e:
            error_trace = traceback.format_exc() 
            print(colored(f"Error in search_web_online: {str(e)}\n{error_trace}",'red'))
            return f"Error in search_web_online: {str(e)}\n{error_trace}"

    async def literature_review(self, query: str, prompt: str) -> str:
        """
        By using this tool, you can search for relevant papers in the literature. Therefore, use your query carefully so that some matched papers could be found, preferably it should not be too specific.
        parameters:
        query: str      # the the query you want to search for in semantic scholar, for relevant papers
        prompt: bool   # promp for the literature reviewer agent
        output: str     # returned result from the literature reviewer agent
        """
        self.no_tool_exec_iters+=1
        try:
            print(colored( f"\n Agent is calling Literature reviewer agent...\nquery: {query}\nprompt: {prompt} ",'blue'))
            response= await self.literature_reviewer.literature_review(query,prompt)
            print(colored(f"Literature review result:\n{response}",'magenta'))
            return response
        except Exception as e:
            error_trace = traceback.format_exc()
            print(colored(f"Error in literature_review: {str(e)}\n{error_trace}",'red'))
            return f"Error in literature_review: {str(e)}"

    async def ask_user(self,question: str) -> str:
        """through this function you can ask the user for more information regarding the epidemic"""
        self.no_tool_exec_iters+=1
        async with print_lock:
            print(colored(f"\nAsking User: {question}...\n", "magenta"))
            return await asyncio.to_thread(
                input,
                colored(f"{question}\n-------------------\n> ", "green")
            )
    def _trim_context_memory(self):
        """
        Trim the memory to remove old messages (action_{t-2},result_{t-2}) reflection_{t-2} )
        """
        try:
            
            if len(self.conv_length)>1 and self.memory:
                print(colored("\nTrimming memory...\n", "yellow"))
                n=self.conv_length[0]
                self.memory = [self.memory[0]] + self.memory[n:]
                self.conv_length=[len(self.memory)]    
                           
        except Exception as e:
            print(colored(f"Error trimming memory: {e}", "red"))
    def _count_tokens(self, message_history):
        self.input_tokens+=message_history.usage().input_tokens
        self.output_tokens+=message_history.usage().output_tokens
        self.tokens+=message_history.usage().total_tokens

if __name__ == "__main__":
   
     
        async def run():    
            prompt="just generate sth dummy without tool use!"
            disc_scientist = DiscoveryScientist(cfg=get_settings(config_path="config.yaml"))
            #math_agent=MathExpert()
            #result=await math_agent.ask_mathematician_agent(prompt)
            result=await disc_scientist.forward(query=prompt)#, scientist_module={"plan":True, "react":True,"reflect":True}, reflection_steps=1,copilot=False)
            #resutl=mathexpert.ask_mathematician_agent(question_prompt)
            print(type(result))
            

        import asyncio
        try:
            x=2/0
        except:
            error_trace = traceback.format_exc() 
            print(colored(f"Error in the main script occured and the simulation failed: {error_trace}", 'red'))
        asyncio.run(run())