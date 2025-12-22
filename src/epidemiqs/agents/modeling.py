import os
import time
import json
import uuid
from datetime import datetime, timezone
import asyncio
from typing import List, Dict, Optional
from os.path import join as ospj
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.tools import Tool
from epidemiqs.agents.prompts import system_prompts
from epidemiqs.utils.utils import log_tokens
from epidemiqs.agents.output_structures import EpidemicModelInfo, NetworkStructure, MechanisticModel, MechanisticModelParams, EpidemicInfo, ReflectResult
from epidemiqs.utils.config import get_settings,Settings
from epidemiqs.utils.long_term_memory import LTM
from epidemiqs.utils.llm_models import choose_model
from epidemiqs.agents.prompts import system_prompts
from termcolor import colored
from epidemiqs.agents.tools import CodeExecutor

class ModelerScientist():
    def __init__(self,llm=choose_model()["scientists"],
                 cfg:Settings=None,
                 name: str="ModelerScientist", 
                 system_prompt:str= system_prompts["modeler_scientist"], 
                 output_type:dataclass =MechanisticModel,
                 no_retries: int = None, 
                 end_strategy: str = "exhaustive",
                 repo:str=ospj(os.getcwd(), "output"),
                 ltm_path:str=None                  
                 ):
        self.name= name
        self.cfg=cfg if cfg is not None else get_settings()
        self.llm=llm if llm is not None else choose_model(self.cfg)["scientists"]
        self.memory = LTM(agent_name=self.name, path=ltm_path).memory if ltm_path is not None else None
        self.ltm_path=ltm_path or ospj("output","LTM.json")
        self.conv_length = []
        self.no_retries= no_retries or self.cfg.workflow.no_retries

        self.react_module = Agent(self.llm,
                                name=name,
                                 system_prompt=system_prompt,
                                 output_type=output_type,
                                 output_retries=self.no_retries,
                                 end_strategy=end_strategy,
                                 tools=[Tool(self.execute_code, takes_ctx=False)],
                                 )
        self.reflect_module = Agent(self.llm,
                                    system_prompt=system_prompt+"Now your only job is to reflect on the generated mechanistic model and make sure that it is accurate and complete; if not, provide your reflection and set the revise_needed=True; O.W., if you are satisfied, set the revise_needed=False.",
                                    output_type=ReflectResult,
                                    output_retries=self.no_retries,
                                    )
        self.plan_module=Agent(self.llm,
                        system_prompt= system_prompt+"Now your only job is to create step by step plan through chain-of-thought to acomplish the task.",)
        
        self.code_executor = CodeExecutor(self.cfg.workflow.time_out_tools)
        self.repo=repo
        self.input_tokens=0
        self.output_tokens=0
        self.tokens=0
        self.no_tool_exec_iters=0
        self.no_reflect_iters=0
        
    async def execute_code(self, code: str, return_vars: List[str] = None,  script_name: str = "model.py") -> str:
        """
        Execute Python code in a persistent environment and return specified variables.
        Args
        code: The Python code to execute.
        return_vars: List of variable names to return after execution.
        script_name: Name of the script file to save the code.
        Returns:
        A string message indicating success or failure and results.
        """
        self.no_tool_exec_iters += 1
        return await self.code_executor.execute(code, return_vars, script_name=script_name,write_to_file=True)
    
    async def forward(self, query) -> MechanisticModel:
            prompt = (
                f" we have following info:Scenario description:  \n ==== {query} \n ======\n"
                f"Please through  chain of thoughts set up an epidemic mechanistic (compartmental) model for the scenario.\n"
                "\nImportant: Your should Solely focus on parametric compartmental model network structure, do not set values for the parameters or initial conditions."
                
            )
            if self.cfg.workflow.scientist_modules.plan:
                think_prompt = (
                    f"Now your only job is to create step by step plan through chain-of-thought to acomplish the task for designing up an epidemic mechanistic model for the scenario.\n"
                    f" we have following info:Scenario description:  \n ==== {query} \n ======\n"
                    "Now please create a step by step plan through chain-of-thought ** Focus exclusively on design of the mechanistic model**\n"
                    "Important: Your should Solely focus on parametric compartmental model network structure, do not set values for the parameters or initial conditions."
                )
                try:
                    plan= await self.plan_module.run(think_prompt)
                    self.log_agent_event("plan", think_prompt, plan.output)
                    self._count_tokens(plan)
                    prompt += plan.output
                    print(colored(f"Plan:\n{plan.output}\n", "blue"))
                except Exception as e:
                    print(colored(f"Error occurred while running plan module: {e}", "red"))
                    self.log_agent_event("plan", think_prompt, error=str(e))
                    plan = None


            result = await self.run_react(prompt)
            for _ in range(self.cfg.workflow.reflection_max_iters):
                if self.cfg.workflow.scientist_modules.reflect:
                    reflection_prompt = (
                        f"Now in this step reflect on the Generated Result: {result}  and make sure that it accurately matches the query: {query}; if not, provide your reflection and set the revise True\n"
                        "otherwise, if you are satisfied with the result, set the revise_needed=False\n"
                    )
                    try:
                        reflection= await self.reflect_module.run(reflection_prompt, message_history=self.memory)
                        print(colored(f"Reflection: {reflection.output.reflection}\nRevise Needed:{reflection.output.revise_needed  }\n", "yellow"))
                        self.log_agent_event("reflection", reflection_prompt, reflection.output.to_dict())
                        self._count_tokens(reflection)
                            
                        if reflection.output.revise_needed:
                            self.no_reflect_iters += 1
                            print(colored(f"Revised Result:\n", "green"))
                            prompt = (
                                f"Now based on your reflection\n {reflection.output}, revise the model and make sure that it is accurate and complete.")
                            result = await self.run_react(prompt)
                            if self.memory and len(self.memory()) > 0:
                                self.conv_length.append(len(self.memory()))
                            self._trim_context_memory()
                        elif not reflection.output.revise_needed:
                            print(colored("No revisions needed based on reflection.", "green"))
                            break  # exit the reflection loop if no revision is needed
                    except Exception as e:
                        print(colored(f"Error occurred while running reflection module: {e}", "red"))
                        self.log_agent_event("reflection", reflection_prompt, error=str(e))
                        reflection = ReflectResult(reflection="", revise_needed=False)
                        
            if self.cfg.workflow.copilot:
                user_feedback="no"
                while user_feedback.lower() not in {"yes","y"}:
                    input_text = colored(f"The generated results are as:\n {result}\n"+"-"*25+f"\nIf you have any comment or suggestion, please provide it here. \nIf the output is good as it is enter \n \"yes\"\n,o.w., Enter your concerns or comments::\n","white")
                    user_feedback=input(input_text)
                    
                    if user_feedback.lower() == "yes":
                        print(colored("Great! No changes needed.", "green"))
                        break
            
                    else:
                        prompt = (
                            f"Now based on the user feedback\n {user_feedback}, revise the model and make sure that it is accurate and complete.")
                        result = await self.run_react(prompt)
                print(colored(f"Final Result:\n{result}\n", "green"))
            try:
                self.total_tokens = self.input_tokens + self.output_tokens
                LTM.save_LTM_direct(agent_name=self.name, agent_memory=self.memory, path=self.ltm_path)
                log_tokens(repo=self.repo,agent_name=self.name,llm_model=self.react_module.model.model_name, input_tokens=self.input_tokens, output_tokens=self.output_tokens, total_tokens=self.tokens)
            except Exception as e:
                print(colored(f"Error logging tokens or saving LTM: {e}", "red"))
            return result
        
        
    async def run_react(self,prompt: str,max_retries: int=3) -> MechanisticModel:
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
                    print(colored(f"Max retries reached. Returning {e} message.", "red"))
                    message=MechanisticModel(name="Modeler Scientist Failed to accomplished the task due to error {e}", compartments=[], transitions={}, epidemic_modeler_reasoning_info=f"Failed after {max_retries} attempts due to error: {e}")
                    self.log_agent_event("react", prompt, message.to_dict(), error=e)
                    
                    return message         
    def _count_tokens(self, message_history):
        self.input_tokens+=message_history.usage().input_tokens
        self.output_tokens+=message_history.usage().output_tokens
        self.tokens+=message_history.usage().total_tokens  
          
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
        

class NetworkScientist():
    def __init__(self,llm=choose_model()["scientists"], 
                 cfg:Settings=None,
                 name: str="NetworkScientist", 
                 system_prompt:str= system_prompts["network_modeler_scientist"],
                 output_type:dataclass =NetworkStructure,
                 no_retries: int = None, 
                 end_strategy: str = "exhaustive",
                 repo:str=ospj(os.getcwd(), "output") ,
                 ltm_path:str=None                  
                 ):
        self.name = name
        self.cfg = cfg if cfg is not None else get_settings()
        self.llm = llm if llm is not None else choose_model(self.cfg)["scientists"]
        self.memory = LTM(agent_name=self.name, path=ltm_path).memory if ltm_path is not None else None
        self.ltm_path = ltm_path or ospj("output","LTM.json")
        self.conv_length = []
        self.no_retries = no_retries or self.cfg.workflow.no_retries
        self.tool_retries=50
        self.react_module = Agent(
            self.llm,
            name=name,
            system_prompt=system_prompt,
            output_type=output_type,
            output_retries=self.no_retries,
            end_strategy=end_strategy,
            retries=self.tool_retries,
            tools=[Tool(self.execute_code, takes_ctx=False)],
        )

        self.reflect_module = Agent(
            self.llm,
            system_prompt=system_prompt+"Now your only job is to reflect on the generated network and make sure that it is accurate and complete; if not, provide your reflection and set the revise_needed=True; O.W., if you are satisfied, set the revise_needed=False.",
            output_type=ReflectResult,
            tools=[Tool(self.execute_code, takes_ctx=False)],
        )

        self.plan_module = Agent(
            self.llm,
            system_prompt=system_prompt+"Now your only job is to create step by step plan through chain-of-thought to acomplish the task.",
        )

        self.code_executor = CodeExecutor(self.cfg.workflow.time_out_tools)
        self.repo = repo
        self.input_tokens = 0
        self.output_tokens = 0
        self.tokens = 0
        self.no_tool_exec_iters = 0
        self.no_reflect_iters = 0


    async def execute_code(self, code: str, return_vars: List[str] = None, script_name: str = "network-design.py") -> str:
        """Execute Python code in a persistent environment."""
        self.no_tool_exec_iters += 1
        result = await self.code_executor.execute(code, return_vars, script_name=script_name, write_to_file=True,timeout=self.cfg.workflow.time_out_tools)
        if self.no_tool_exec_iters > self.tool_retries-5:
            message=f"\nNote: You have executed more than {self.no_tool_exec_iters} code blocks, You have {self.tool_retries - self.no_tool_exec_iters} more tries left before reaching the maximum limit of {self.tool_retries} code executions. Make sure to finalize your answer ASAP."
            result += message
        print(colored(f"Code Execution Result:\n{result}\n", "green"))
        return  result


    def _trim_context_memory(self):
        """Trim the memory to remove old messages (action_{t-2},result_{t-2}) reflection_{t-2} )"""
        
        try:
            if len(self.conv_length) > 1 and self.memory:
                print(colored("\nTrimming memory...\n", "yellow"))
                n = self.conv_length[0]
                self.memory = [self.memory[0]] + self.memory[n:]
                self.conv_length = [len(self.memory)]
        except Exception as e:
            print(colored(f"Error trimming memory: {e}", "red"))


    async def run_react(self, prompt: str, max_retries: int = 3):
        """Unified react execution with retries, token counting, logging."""
        for attempt in range(max_retries):
            try:
                result = await self.react_module.run(prompt, message_history=self.memory)
                self.memory = result.all_messages()

                # log
                try:
                    self.log_agent_event("react", prompt, result.output.to_dict())
                except:
                    pass

                print(colored(f"{self.name} result:\n{result.output}\n", "green"))
                self._count_tokens(result)

                return result.output

            except Exception as e:
                if attempt == max_retries - 1:
                    print(colored(f"Max retries reached. Returning error: {e}", "red"))
                    message = NetworkStructure(
                        network_details=f"Network Science Failed to address the query due to error: {e}",
                        network_paths="",
                        plot_paths={},
                        reasoning_info=f"Failed after {max_retries} attempts due to error: {e}"
                    )
                    try:
                        self.log_agent_event("react", prompt, message.to_dict(), error=e)
                    except:
                        pass
                    return message


    def _count_tokens(self, message):
        """Accumulate input/output tokens from usage."""
        try:
            self.input_tokens += message.usage().input_tokens
            self.output_tokens += message.usage().output_tokens
            self.tokens += message.usage().total_tokens
        except:
            pass


    def log_agent_event(self,
        module_name: str,
        input_data=None,
        output_data=None,
        error=None,
        meta: dict | None = None,
        log_path: str = None
    ) -> Dict:

        log_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "module": module_name,
            "input": input_data,
            "output": output_data,
            "error": str(error) if error else None,
            "no_exec_iter": self.no_tool_exec_iters,
            "no_reflections": self.no_reflect_iters,
            "meta": meta or {}
        }

        log_path = log_path if log_path is not None else ospj(self.repo, f"{self.name}.jsonl")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return log_entry


    async def forward(self, query) -> NetworkStructure:
        is_epidemic_info = isinstance(query, EpidemicInfo)

        prompt = (
            f" we have following info from  Discovery phase:{query.description if is_epidemic_info else query}\n"
            f"and gathered information about contact network structure: {query.contact_network_structure if is_epidemic_info  else ''}\n"
            +(f"in case it might be helpful, the overall task that we need you to design the network is  {query.task}" if is_epidemic_info else '')
            +(f"and data path prvided by user: {query.data_paths}" if is_epidemic_info and query.data_paths else '')
            +(f"and data regarding network is provided by user: {query.network_path}" if is_epidemic_info and query.network_path else '')
            +f"""Please Use the following format of ReAct (Reasoning, Action, and Observation) 
                you are in one of the following steps, act accordingly:
                    - Reason: what is the population structure that the disease is spreading over, and what steps(actions) I need to take to create a network that best represent the population structure.
                    - Action: (through executing python code): write python code to construct the contact network structure using the relevant informtion
                    - Observe: Reflect on the result from your action, and answer the question:  Is this the best contact network structure for the scenario? what are the mismatches? how to improve it?
                    Repeat this process until you are satisfied with the network structure. \n
                    Final Think: Now the network is exaclty what I want, and I am satisfied with the network structure, it is accurate and reflects the contact strucutre of the population that disease spread over.
                Final answer:  the contact network structure that best represent the population structure and satisfy all the metric, save the network in a file and return the path to the file, and ruturn your reasoning and logic for the network design.
                """
        )

        #  PLAN STEP 
        if self.cfg.workflow.scientist_modules.plan:
            think_prompt = (
                f"Now your only job is to create step by step plan through chain-of-thought to acomplish the task for design and creating a network that best represent the population structure..\n"
                f" we have following info from discovery phase as :{query.description if is_epidemic_info else query}\n"
                +(
                    f"which is gathered and the information or suggestions for network structure as: {query.contact_network_structure}"
                    if isinstance(query, EpidemicInfo)
                    else ''
                )
                +(f"in case it might be helpful, the overall task that we need you to design the network is  {query.task}" if is_epidemic_info else '')
                +f"Network structure: {query.contact_network_structure if is_epidemic_info else ''}\n"
                "Now please create a step by step plan through chain-of-thought and ** Focus exclusively on the design and construction of the network** that best represent the population structure. \n"
            )

            try:
                plan = await self.plan_module.run(think_prompt)
                print(colored(f"Plan:\n{plan.output}\n", "blue"))
                self.log_agent_event("plan", think_prompt, plan.output)
                self._count_tokens(plan)
                act_prompt = prompt + plan.output
                result = await self.run_react(act_prompt)

            except Exception as e:
                print(colored(f"Error in plan step: {e}", "red"))
                self.log_agent_event("plan", think_prompt, error=str(e))
                plan = None
                result = await self.run_react(prompt)

        else:
            result = await self.run_react(prompt)

        print(colored(f"Result:\n{result}\n", "green"))

        # REFLECTION LOOP
        for _ in range(self.cfg.workflow.reflection_max_iters):
            if self.cfg.workflow.scientist_modules.reflect:

                reflection_prompt = (
                    f"Now in this step only reflect on the generated results:\n {result} and make sure that it is accurate and complete and the generated network matches the criteria in query{prompt}; if not, provide your reflection and provide feedback how to correct it and set the revise True; O.W., if you are satisfied, set the revise_needed=False  \n"
                )

                try:
                    reflection = await self.reflect_module.run(reflection_prompt, message_history=self.memory)
                    print(colored(f"Reflection\n: {reflection.output.reflection}", "yellow"))
                    self.log_agent_event("reflection", reflection_prompt, reflection.output.to_dict())
                    self._count_tokens(reflection)

                except Exception as e:
                    print(colored(f"Error in reflection module: {e}", "red"))
                    self.log_agent_event("reflection", reflection_prompt, error=str(e))
                    reflection = ReflectResult(reflection="", revise_needed=False)

                if reflection.output.revise_needed:
                    self.no_reflect_iters += 1
                    revise_prompt = (
                        f"Now based on your reflection\n {reflection.output}, revise the network and make sure that it is accurate and complete."
                    )
                    result = await self.run_react(revise_prompt)
                    print(colored(f"Revised Result\n: {result}\n", "green"))

                    if self.memory and len(self.memory) > 0:
                        self.conv_length.append(len(self.memory))
                    self._trim_context_memory()

                else:
                    print(colored("No revisions needed based on reflection.", "green"))
                    break

        # COPILOT FEEDBACK 
        if self.cfg.workflow.copilot:
            user_feedback = "no"
            while user_feedback.lower() not in {"yes","y"}:
                input_text = colored(
                    f"\nThe generated results are as:\n"
                    + "-"*25
                    + f"{result}\n"
                    + "-"*25
                    + "If you have any comment or suggestion, please provide it here.\n"
                      "If the output is good as it is enter \n \"yes\"\n,o.w.,\n Enter your concerns or comments::\n",
                    "white"
                )
                user_feedback = input(input_text)

                if user_feedback.lower() in {"yes","y"}:
                    print(colored("\nGreat! No changes needed.", "green"))
                else:
                    revise_prompt = (
                        f"Now based on the user feedback\n {user_feedback}, revise the network and make sure that it is accurate and complete."
                    )
                    print(colored(f"Revised Result:\n", "green"))
                    result = await self.run_react(revise_prompt)
                    

        #  SAVE MEMORY + TOKEN LOGGING 
        try:
            self.total_tokens = self.input_tokens + self.output_tokens
            LTM.save_LTM_direct(agent_name=self.name, agent_memory=self.memory, path=self.ltm_path)

            log_tokens(
                repo=self.repo,
                agent_name=self.name,
                llm_model=self.react_module.model.model_name,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                total_tokens=self.tokens
            )
        except Exception as e:
            print(colored(f"Error logging tokens or saving LTM: {e}", "red"))

        return result

class ParameterScientist():
    def __init__(self,llm=None,
                 cfg:Settings=None,
                 name: str="ParameterScientist",
                 system_prompt:str= system_prompts["parameter_setter_scientist"], 
                 output_type:dataclass =MechanisticModelParams,
                 no_retries: int = None, 
                 end_strategy: str = "exhaustive",
                 repo:str=ospj(os.getcwd(), "output"),
                 ltm_path: str = None
                 ):

        self.name= name
        self.cfg=cfg if cfg is not None else get_settings()
        self.llm=llm if llm is not None else choose_model(self.cfg)["scientists"]
        self.memory = LTM(agent_name=self.name, path=ltm_path).memory if ltm_path is not None else None
        self.ltm_path=ltm_path or ospj("output","LTM.json")
        self.conv_length = []
        self.no_retries= no_retries or self.cfg.workflow.no_retries

        self.react_module = Agent(
            self.llm,
            name=name,
            system_prompt=system_prompt,
            output_type=output_type,
            output_retries=self.no_retries,
            end_strategy=end_strategy,
            tools=[Tool(self.execute_code, takes_ctx=False)],
        )
        
        self.reflect_module = Agent(
            self.llm,
            system_prompt=system_prompt,
            output_type=ReflectResult,
        )
        
        self.plan_module = Agent(
            self.llm,
            system_prompt=system_prompt + 
                "Now your only job is to create step by step plan through chain-of-thought to acomplish the task.",
        )

        self.input_tokens = 0
        self.output_tokens = 0
        self.tokens = 0
        self.no_tool_exec_iters = 0
        self.no_reflect_iters = 0

        self.code_executor = CodeExecutor(self.cfg.workflow.time_out_tools)
        self.repo=repo


    async def execute_code(self, code: str, return_vars: List[str] = None, script_name: str = "paramters-setting.py") -> str:            
        self.no_tool_exec_iters += 1
        
        result = await self.code_executor.execute(
            code,
            return_vars,
            script_name=script_name,
            write_to_file=True
        )
        return result


    # TOKEN COUNTING
    def _count_tokens(self, message):
        try:
            self.input_tokens += message.usage().input_tokens
            self.output_tokens += message.usage().output_tokens
            self.tokens += message.usage().total_tokens
        except:
            pass


    # LOGGING
    def log_agent_event(
        self,
        module_name: str,
        input_data=None,
        output_data=None,
        error=None,
        meta: dict | None = None,
        log_path: str = None
    ) -> Dict:

        log_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "module": module_name,
            "input": input_data,
            "output": output_data,
            "error": str(error) if error else None,
            "no_exec_iter": self.no_tool_exec_iters,
            "no_reflections": self.no_reflect_iters,
            "meta": meta or {}
        }

        log_path = log_path if log_path is not None else ospj(self.repo, f"{self.name}.jsonl")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return log_entry


    # REACT RUNNER WITH RETRIES
    async def run_react(self, prompt: str, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                result = await self.react_module.run(prompt, message_history=self.memory)
                self.memory = result.all_messages()

                # logging
                try:
                    self.log_agent_event("react", prompt, result.output.to_dict())
                except:
                    pass

                print(colored(f"{self.name} result:\n{result.output}\n", "green"))
                self._count_tokens(result)
                return result.output

            except Exception as e:
                if attempt == max_retries - 1:
                    print(colored(f"Max retries reached. Returning error: {e}", "red"))
                    message = MechanisticModelParams(
                        params_details=f"ParameterScientist failed to address the query due to error: {e}",
                        initial_conditions="",
                        reasoning_info=f"Failed after {max_retries} attempts due to error: {e}"
                    )
                    try:
                        self.log_agent_event("react", prompt, message.to_dict(), error=e)
                    except:
                        pass
                    return message


    # MEMORY TRIMMING
    def _trim_context_memory(self):
        """Trim the memory to remove old messages (action_{t-2},result_{t-2}) reflection_{t-2} )"""
        
        try:
            if len(self.conv_length) > 1 and self.memory:
                print(colored("\nTrimming memory...\n", "yellow"))
                n = self.conv_length[0]
                self.memory = [self.memory[0]] + self.memory[n:]
                self.conv_length = [len(self.memory)]
        except Exception as e:
            print(colored(f"Error trimming memory: {e}", "red"))


    # FORWARD plan/react/reflect loop
    async def forward(self, query, network:NetworkStructure=None,
                      model:MechanisticModel=None) -> MechanisticModelParams:
        is_network_structure = isinstance(network, NetworkStructure)
        is_model_mechanistic_model = isinstance(model, MechanisticModel)
        is_epidemic_info = isinstance(query, EpidemicInfo)
        excluded_keys = ['plot_paths',"reasoning_info"] if is_epidemic_info else []
        network_details = {k: v for k, v in network.to_dict().items() if k not in excluded_keys} if is_network_structure else network    


        prompt = (
            (f"we have following info from  Discovery phase:{query.description}\n" if is_epidemic_info else f"we have following info: {query}\n")
            +(f"and disease  R0: {query.R_0}\n" if is_epidemic_info and query.R_0 else "")
            +(f"and analytical insights: {query.math_solution}\n" if is_epidemic_info else "")
            +(f"and data path prvided by user: {query.data_paths}" if is_epidemic_info and query.data_paths else '')
            +(f" we have created the network with Network specification:\n {network_details}\n" if is_network_structure else '')
            +(f" and designed the Epidemic model: {model}\n"+"-"*25+"\n" if is_model_mechanistic_model else '')
            #+f"""Please Use the following format of ReAct (Reasoning, Action, and Observation) 
            #    you are in one of the following steps, act accordingly:
            #        - Reason: what is the population structure that the disease is spreading over, and what steps(actions) I need to take to set the parameters for the mechanistic model that best represent the scenario.
            #        - Action: (write python code): write python code to set the parameters for mechanistic model using the relevant informtion
            #        - Observe: Reflect on the result from your action, and answer the question:  Is this the best parameters for the mechanistic model for the scenario? what are the mismatches? how to improve it?
            #       Repeat this process until you are satisfied with the parameters. \n
            #        Final Think: Now the parameters are exaclty what I want, and I am satisfied with them,
            #        now return the parameters and initial conditions for the mechanistic model that best represent the scenario, and ruturn your reasoning and logic for the parameters setting.
          #      """
        )

        # PLAN STEP
        if self.cfg.workflow.scientist_modules.plan:

            think_prompt = (
                f"Now your only job is to create step by step plan through chain-of-thought to acomplish the task for setting the parameters for mechanistic model that best represent the scenario..\n"
                f" we have following info from discovery phase as :{query}\n"
                +f" we have created the network with Network specification:\n {network_details}\n"
                f" and designed the Epidemic model: {model}\n"+"-"*25+"\n"
                "Now please create a step by step plan through chain-of-thought **Focus exclusively on how to set paramters for mechanistic model** that best represent the scenario. your output should only be the plan to determine what to do.\n"
            )

            try:
                plan = await self.plan_module.run(think_prompt)
                self.log_agent_event("plan", think_prompt, plan.output)
                print(colored(f"Plan: {plan.output}", "blue"))
                self._count_tokens(plan)
                result = await self.run_react(prompt + plan.output)

            except Exception as e:
                print(colored(f"Error in plan step: {e}", "red"))
                self.log_agent_event("plan", think_prompt, error=str(e))
                result = await self.run_react(prompt)

        else:
            result = await self.run_react(prompt)


        print(colored(f"Result:\n{result}\n", "green"))

        # REFLECTION LOOP
        for _ in range(self.cfg.workflow.reflection_max_iters):
            if self.cfg.workflow.scientist_modules.reflect:

                reflection_prompt = (
                    f"Now in this step only reflect on the generated results {result} based on input query {prompt} and make sure that it is accurate and complete; if not, provide your reflection and provide feedback how to correct it and set the revise_needed True; O.W., if you are satisfied, set the revise_needed=False  \n"
                )

                try:
                    reflection = await self.reflect_module.run(reflection_prompt, message_history=self.memory)
                    print(colored(f"Reflection: {reflection.output.reflection}", "yellow"))
                    self.log_agent_event("reflection", reflection_prompt, reflection.output.to_dict())
                    self._count_tokens(reflection)

                except Exception as e:
                    print(colored(f"Error in reflection module: {e}", "red"))
                    self.log_agent_event("reflection", reflection_prompt, error=str(e))
                    reflection = ReflectResult(reflection="", revise_needed=False)

                if reflection.output.revise_needed:
                    self.no_reflect_iters += 1
                    revise_prompt = (
                        f"Now based on your reflection\n {reflection.output.reflection}, revise the results and make sure that it is accurate and complete."
                    )
                    print(colored(f"Revised Result:\n", "green"))
                    result = await self.run_react(revise_prompt)
                    

                    if self.memory and len(self.memory) > 0:
                        self.conv_length.append(len(self.memory))

                    self._trim_context_memory()

                else:
                    print(colored("No revisions needed based on reflection.", "green"))
                    break


        # COPILOT FEEDBACK
        if self.cfg.workflow.copilot:
            user_feedback = "no"
            while user_feedback.lower() not in {"yes","y"}:
                input_text = colored(
                    f"\nThe generated results are as:\n"
                    + "-"*25
                    + f"{result}\n"
                    + "-"*25
                    + "If you have any comment or suggestion, please provide it here.\n"
                      "If the output is good as it is enter \n \"yes\"\n,o.w., Enter your concerns or comments::\n",
                    "white"
                )
                user_feedback = input(input_text)

                if user_feedback.lower() in {"yes","y"}:
                    print(colored("\nGreat! No changes needed.", "green"))
                else:
                    revise_prompt = (
                        f"Now based on the user feedback\n {user_feedback}, revise the parameters and make sure that it is accurate and complete."
                    )
                    result = await self.run_react(revise_prompt)
                    print(colored(f"Revised Result:\n{result}\n", "green"))


        try:
            self.total_tokens = self.input_tokens + self.output_tokens

            LTM.save_LTM_direct(
                agent_name=self.name,
                agent_memory=self.memory,
                path=self.ltm_path
            )

            log_tokens(
                repo=self.repo,
                agent_name=self.name,
                llm_model=self.react_module.model.model_name,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                total_tokens=self.tokens
            )
        except Exception as e:
            print(colored(f"Error logging tokens or saving LTM: {e}", "red"))

        return result   
                                 
class ModelingTeam():
    def __init__(self,
                llm=None,
                cfg:Settings=None,
                phase:str="Modeling",
                name: str="ModelingTeam",
                output_type:dataclass = EpidemicModelInfo,
                ltm_path:str=None):
        
        self.network_scientist = NetworkScientist(llm=llm,cfg=cfg,ltm_path=ltm_path)
        self.model_scientist = ModelerScientist(llm=llm,cfg=cfg,ltm_path=ltm_path)
        self.parameter_scientist = ParameterScientist(llm=llm,cfg=cfg,ltm_path=ltm_path)
        self.output_type = output_type
        self.name = name
        self.phase = phase
        self.tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
    
        
        print(colored(f"Modeling Team initialized with agents:\n{self.network_scientist.react_module.name} powered by: {self.network_scientist.react_module.model.model_name},\n{self.model_scientist.react_module.name} powered by:{self.model_scientist.react_module.model.model_name},\n{self.parameter_scientist.react_module.name} powered by {self.parameter_scientist.react_module.model.model_name}", "light_cyan"))
    async def forward(self,query)-> EpidemicModelInfo:  # Changed to accept dict instead of str
            """A high-level function to run the modeling team agents as: concurrent: model and network design -> hand-off to parameter setting."""
            st=time.time()
            network, model=await asyncio.gather(
                self.network_scientist.forward(query),
                self.model_scientist.forward(query)
            )
            parameters= await self.parameter_scientist.forward(query, network=network, model=model)
            self.input_tokens=self.network_scientist.input_tokens + self.model_scientist.input_tokens + self.parameter_scientist.input_tokens
            self.output_tokens=self.network_scientist.output_tokens + self.model_scientist.output_tokens + self.parameter_scientist.output_tokens
            self.tokens=self.network_scientist.tokens + self.model_scientist.tokens+ self.parameter_scientist.tokens
            log_tokens(repo=self.network_scientist.repo,csv_name="tokens_by_phase.csv",agent_name=self.phase,
                       llm_model=self.network_scientist.react_module.model.model_name, input_tokens=self.input_tokens, 
                       output_tokens=self.output_tokens,
                       total_tokens=self.tokens,time=time.time()-st)
            print(colored(f"\nTotal Tokens used by {self.name}: {self.tokens}\n","red"))
            modeling_output = self.output_type(
                network_details=network.network_details,
                network_paths=network.network_paths,
                plot_paths=network.plot_paths,
                network_reasoning_info=network.reasoning_info,
                model_name=model.name,
                compartments=model.compartments,
                transitions=model.transitions,
                epidemic_modeler_reasoning_info=model.reasoning_info,
                parameters=parameters.parameters,
                initial_condition_desc=parameters.initial_condition_desc,
                initial_conditions=parameters.initial_conditions,
                params_reasoning_info=parameters.reasoning_info
            )
            print(colored(modeling_output, "cyan"))  # Adding a color for clarity
            return modeling_output        

  
if __name__ == "__main__":
    
    async def main(scenario=None):
        with open("output/log.json", "r") as f:
            log=json.loads(f.read())
        data=(log["Discovery"]["iteration_0"][0]["data"])
        epidemic_info=EpidemicInfo(**data)
        print(f"Epidemic Info:\n {epidemic_info}")
        modeling_team = ModelingTeam(cfg=get_settings(config_path="config.yaml"))
        #model=await modeling_team.forward(query="Model sth nice for for two competing exclusive viruss over small city of 100 people ( where some people do not talk other people at all), ensure both virus can spread meaning tau is bigger than /lambda1 ( each infectected can become susceptible)" )
        result=await modeling_team.forward(query =epidemic_info)#,network=None, model=None) 
        print(f"Final Result:\n {result}")
    import asyncio
    
    asyncio.run(main())
