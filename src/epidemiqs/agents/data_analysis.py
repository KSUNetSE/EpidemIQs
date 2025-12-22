import os
import time
import json
import uuid
from datetime import datetime, timezone
from termcolor import colored
from dataclasses import dataclass, asdict
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from typing import List, Dict, Optional, Union  
from termcolor import colored, cprint
from epidemiqs.agents.tools import CodeExecutor
from pydantic_ai.tools import Tool
from os.path import join as ospj
from epidemiqs.agents.output_structures import SimulationResult,EpidemicInfo, AnalysisResult, ReflectResult,AnalysisResponse
import traceback
import time 
from epidemiqs.utils.llm_models import choose_model
from epidemiqs.utils.config import get_settings, Settings
from epidemiqs.utils.utils import log_tokens
from epidemiqs.agents.task_expert_agents import VisionAgent
from epidemiqs.agents.task_expert_agents import DataExpert
from epidemiqs.utils.long_term_memory import LTM
from epidemiqs.agents.prompts import system_prompts
cfg=get_settings()


@dataclass  
class AnalysisResponse:
    results_analysis: List[str] # the analysis of results of simulations if multiple is done, include all. Also, including the metrics you have extracted from the data and the image, and 
    metric_table: str # table in latex format that contains the metrics for all simulation results
    evlauation_reasoning_info:str # and you must give the reasons you have to justify your decisions such choosing metrics, costs, evaluations etc. against hypthetic criticism of why  these are the best choices.\n 

    def to_dict(self):
        return self.__dict__.copy()
    
    def to_json(self,iteration=0):
        path=os.path.join("output",f"{"data_analysis"}_info.json")
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
 

@dataclass
class ReflectResponse:
    """Class to hold search results."""
    reflection: str
    revise_needed: bool

    def to_dict(self):
        return self.__dict__.copy()
@dataclass
class Mitigation_Response:
    """class for decision on whether mitigation is needed or not."""
    further_mitigation_needed: bool
    mitigation_reasoning_info:str # and you must give the reasons you have to justify your decisions such choosing metrics, costs, evaluations etc. against hypthetic criticism of why  these are the best choices.\n

    def to_dict(self):
        return self.__dict__.copy()
    
@dataclass
class AnalysisResult:
    """Class to hold evaluation results."""
    result_analysis: str
    metric_table: str
    mitigation_needed:bool
    evaluation_reasoning_info:str
    mitigation_reasoning_info:str
    def to_dict(self):
        return self.__dict__.copy()
    

class AnalysisTeam: 
    def __init__(self,llm=None,
                 cfg:Settings=None,
                 phase:str="Analysis",
                 name: str="DataScientist",
                 system_prompt:str= system_prompts["data_scientist"], 
                 scientist_output_type:dataclass=AnalysisResponse,
                 output_type:dataclass=AnalysisResult,
                 no_retries: int =None, 
                 end_strategy: str = "exhaustive",
                 repo:str=ospj(os.getcwd(), "output"),
                 ltm_path:str=None
                 ):
        self.name = name
        self.cfg = cfg if cfg is not None else get_settings()
        self.llm = llm if llm is not None else choose_model(self.cfg)["scientists"]
        self.memory = LTM(agent_name=self.name, path=ltm_path).memory if ltm_path is not None else None
        self.conv_length = []
        self.ltm_path = ltm_path or ospj("output","LTM.json")
        self.no_retries = no_retries or self.cfg.workflow.no_retries

        self.tool_retries = 50
        self.react_module = Agent(
            self.llm,
            name=name, 
            system_prompt=system_prompt,
            output_type=scientist_output_type,
            output_retries=self.no_retries,
            retries=self.tool_retries,
            end_strategy=end_strategy,
            tools=[
                Tool(self.talk_to_data_expert, takes_ctx=False),
                Tool(self.talk_to_vision_expert, takes_ctx=False),
                #Tool(execute_code,takes_ctx=False),
            ]
        )

        self.reflect_module = Agent(
            self.llm,
            name="reflect-module",
            system_prompt=system_prompt+"\n Now I want you to exclusively focus on reflecting on the generated response and check if the was accurate, include all user requests and ensure there was no deviation from the user request. ",
            output_type=ReflectResult,
            output_retries=self.no_retries,
        )
        
        self.plan_module = Agent(
            self.llm,
            name="plan-module",
            system_prompt=system_prompt+"\n Now you should exclusively focus on thinking and planning the steps you need to take to accomplish the task, use chain-of-thought to think about what are what are the relavenat measures and insights and how you should extract them simulation results.",
        )
        
        self.vision_expert = VisionAgent(cfg=self.cfg)
        self.data_expert = DataExpert(cfg=self.cfg) 
        self.repo = repo

        # token + iteration counters (same pattern as NetworkScientist)
        self.input_tokens = 0
        self.output_tokens = 0
        self.tokens = 0
        self.no_tool_exec_iters = 0
        self.no_reflect_iters = 0

        self.code_executor = CodeExecutor(default_timeout=self.cfg.workflow.time_out_tools)   
        self.phase = phase 
        print(colored(
            f"Analysis Team initialized with agents:\n"
            f"{self.name} is powered by {self.react_module.model.model_name}\n"
            f"{self.vision_expert.name} is powered by {self.vision_expert.react_module.model.model_name}\n"
            f"{self.data_expert.name} is powered by {self.data_expert.react_module.model.model_name}",
            'green'
        ))

    def _trim_context_memory(self):
        """Trim memory to remove old messages."""
        try:
            if len(self.conv_length) > 1 and self.memory:
                print(colored("\nTrimming memory...\n", "yellow"))
                n = self.conv_length[0]
                self.memory = [self.memory[0]] + self.memory[n:]
                self.conv_length = [len(self.memory)]
        except Exception as e:
            print(colored(f"Error trimming memory: {e}", "red"))

    def _count_tokens(self, message):
        """Accumulate input/output tokens from usage."""
        try:
            self.input_tokens += message.usage().input_tokens
            self.output_tokens += message.usage().output_tokens
            self.tokens += message.usage().total_tokens
        except:
            pass

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

    async def run_react(self, prompt: str, max_retries: int = 3) -> AnalysisResponse:
        """Unified react execution with retries, token counting, logging."""
        for attempt in range(max_retries):
            try:
                result = await self.react_module.run(user_prompt=prompt, message_history=self.memory)
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
                    # fallback AnalysisResponse
                    message = AnalysisResponse(
                        results_analysis=f"Analysis failed to address the query due to error: {e}",
                        metric_table="",
                        evlauation_reasoning_info=f"Failed after {max_retries} attempts due to error: {e}"
                    )
                    try:
                        self.log_agent_event("react", prompt, message.__dict__, error=e)
                    except:
                        pass
                    return message

    async def forward(self, query, simulation:SimulationResult) -> AnalysisResult:
        """Evaluating the  results of the simulation by extracting statistical metrics from the data."""
        st = time.time()
        is_epidemic_info = isinstance(query, EpidemicInfo)
        is_simulation_result = isinstance(simulation, SimulationResult)
        print(is_epidemic_info, is_simulation_result)
        plan_prompt = (
                f"The description for the initial query is: {query.description if is_epidemic_info else str(query)}.\n"
                +(f"(The user initial task was : {query.task}\n" if is_epidemic_info else '')
                +(f"and data path provided by user: {query.data_paths}" if is_epidemic_info and query.data_paths else '')
                +(f"""and the simulation results are as follows:\n {simulation.simulation_details}\nplot paths at:{simulation.plot_path}\nnumerical data at{simulation.stored_result_path}\n and Simulation logic as:{simulation.reasoning_info}""" if is_simulation_result else str(simulation))
                +"""
                
                Now,I want to analyze the simulation results and provide a comprehensive analysis of the results, including the metrics you have extracted from the data and the image.
                You have two Expert Agents to assist you:
                1. Data Expert: This agent can extract numerical data from files.
                2. Vision Expert: This agent has vision capabilty and can analyze images and provide insights based on the visual data.
                Please use chain-of-thought reasoning to think about how and the steps you need to take to accomplish the task by *Focus exclusively on Analysis* and *Evaluation* of the simulation results.
                """)
        prompt = (
                f"The description for the initial query is: {query.description if is_epidemic_info else str(query)}.\n"
                +(f"and data path provided by user: {query.data_paths}" if is_epidemic_info and query.data_paths else '')
                +(f"and the simulation results are as follows:\n {simulation.to_dict()}\n" if is_simulation_result else str(simulation))
                +"""Now please perform the analysis of the simulation results and provide a comprehensive analysis of the results, including the metrics you have extracted from the data and the image.
                You have two Expert Agents to assist you:
                1. Data Expert: This agent can extract numerical data from files.
                2. Vision Expert: This agent has vision capabilty and can analyze images and provide insights based on the visual data.
                Please always reflect on the results and check if the results make sense, if there are contradictions in the data, plan and redo the process or mention that in your response.
                """)

        # PLAN STEP
        if self.cfg.workflow.scientist_modules.plan:
            print(colored(f"\nThinking about the analysis of the simulation results...\n","blue"))
            try:
                plan = self.think_agent = await self.plan_module.run(user_prompt=plan_prompt)
                self._count_tokens(plan)
                print(colored(f"Think Agent Response:\n {self.think_agent.output}","blue"))
                self.log_agent_event("plan", plan_prompt, plan.output)

                act_prompt = prompt + f"\nThe suggested plant is:\n{str(plan.output)}"
                result = await self.run_react(act_prompt)

            except Exception as e:
                print(colored(f"Error in plan step: {e}", "red"))
                self.log_agent_event("plan", plan_prompt, error=str(e))
                plan = None
                result = await self.run_react(prompt)
        else:
            result = await self.run_react(prompt)
        
        print(colored(f"\nEvaluation Agent Response:\n {result}","green"))

        # REFLECTION LOOP
        for _ in range(self.cfg.workflow.reflection_max_iters):
            reflect_prompt = f"""Now I want you to reflect on the generated response\n {result}\n  through Chain-of-thought and check if they are accurate and comprehensivly address all required fields based initial query\n{query}\n.
                    Make sure that all measures are included, accurate, and make sense , and units are provided for each metric.
                    ensure results from the vision agent and the data extractor align with each other.
                    If there are mistakes, inaccuracies, hallucination or any tasks missed, please provide accurate instructions in your reflection to correct the answer; OW.; if you are 
                    satisified with results, then no revision is needed and set revised_needed=False\n"""
                    
            if self.cfg.workflow.scientist_modules.reflect:
                print(colored(f"\nReflecting on the evaluation results...\n","yellow"))
                try:
                    reflection = await self.reflect_module.run(user_prompt=reflect_prompt, message_history=self.memory)
                    self._count_tokens(reflection)
                    print(colored(f"Reflection Agent Response: {reflection.output}","white"))
                    try:
                        self.log_agent_event("reflection", reflect_prompt, reflection.output.to_dict())
                    except:
                        pass
                except Exception as e:
                    print(colored(f"Error in reflection module: {e}", "red"))
                    self.log_agent_event("reflection", reflect_prompt, error=str(e))
                    reflection = ReflectResult(reflection="", revise_needed=False)

                if reflection.output.revise_needed:
                    self.no_reflect_iters += 1
                    revise_prompt = "Please based on the reflection provided take action to complete the task accurately:\n {reflection.output.ReflectionDetails}"
                    result = await self.run_react(revise_prompt)
                    print(colored(f"Revised Evaluation Agent Response: {result}","green"))

                    if self.memory and len(self.memory) > 0:
                        self.conv_length.append(len(self.memory))
                    self._trim_context_memory()
                else:
                    print(colored("No revision needed based on reflection.","green"))
                    break

        # COPILOT FEEDBACK 
        if self.cfg.workflow.copilot:
            user_feedback = "no"
            while user_feedback.lower() not in ['yes','y']:
                input_text = colored(
                    f"\nThe generated results are as:\n"
                    + "-"*25
                    + f"{result}\n"
                    + "-"*25
                    + "If you have any comment or suggestion, please provide it here.\n If the output is good as it is enter \n \"yes\"\n,o.w.,\n Enter your concerns or comments:\n",
                    "white"
                )
                user_feedback = input(input_text)
                if user_feedback.lower() in ["yes", "y"]:
                    print(colored("\nGreat! No changes needed.", "green"))
                    break
                else:
                    revise_prompt = (
                        f"Now based on the user feedback\n {user_feedback}, address the concerns by incorporating the feedbackf for simulations\n"
                    )
                    result = await self.run_react(revise_prompt)
                    print(colored(f"Revised Result:\n{result}\n", "green"))

        print(colored(f"Analysis Team Results: {result}","green"))

        # SAVE MEMORY + TOKEN LOGGING 
        try:
            self.total_tokens = self.input_tokens + self.output_tokens
            LTM.save_LTM_direct(agent_name=self.name, agent_memory=self.memory, path=self.ltm_path)

            log_tokens(
                repo=self.repo,
                agent_name=self.react_module.name,
                llm_model=self.react_module.model.model_name,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                total_tokens=self.tokens
            )
            log_tokens(
                repo=self.repo,
                agent_name=self.phase,
                csv_name="tokens_by_phase.csv",
                llm_model=self.react_module.model.model_name,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                total_tokens=self.tokens,
                time=time.time()-st
            )
        except Exception as e:
            print(colored(f"Error logging tokens or saving LTM: {e}", "red"))

        return AnalysisResult(
            result_analysis=result.results_analysis,
            metric_table=result.metric_table,
            mitigation_needed=False,
            evaluation_reasoning_info=result.evlauation_reasoning_info,
            mitigation_reasoning_info="Not Relevant to the case"  # mitigation.output.mitigation_reasoning_info
        )

    async def talk_to_data_expert(self, query:str,data_paths:List[str]) -> str:  
        """this tool is  useful to  ask for extracting  data from the saved csv file  that stored the simualtion resultsand provide the required information you like to see by calling this function."""
        
        print(colored( f"\nTalking to Data Expert..\ninquiry: {query}+ \ndata_path:{str(data_paths)} ",'blue'))
        self.no_tool_exec_iters += 1
        try:
            result = await self.data_expert.forward(query=query, data_paths=data_paths)
            print(colored(f"\nDataExpert Response {"-"*20}:\n{result} \n",'white'))
            return (
                "–" * 5 + "\n"
                + "DataExpert Response:\n"
                + str(result) + "\n"
                + "–" * 5 + "\n"
                + "Ensure to verify whether these results accurately represent the data and there is no contradictions."
            )
        except Exception as e:  
            error_trace = traceback.format_exc()
            print(colored(f"Extractor Agent Could not respond: {str(error_trace)}","red"))
            return f"DataExpert Is Not Accessibe at the moment,\n{error_trace}"
    
    async def talk_to_vision_expert(self, query: str, image_paths:List[str]) -> str:
        """
        Use tool to have conversation with vision agent to analyze the plotted results of the simulation. 
        saved plotted results  as population of each state (numbers) over time(day)."""

        print(colored( f"\n{self.name} talking to vision agent.\n inquiry:{query} \n at path:{image_paths}","green"))
        self.no_tool_exec_iters += 1

        try:
            result_vision = await self.vision_expert.forward(query=query, image_paths=image_paths)
            print(colored(f"Vision Agent Response:{result_vision}","white"))
            return (
                f"\n{'–'*5}\n"
                f"Vision Expert Response:\n{result_vision}\n"
                f"{'–'*5}\n"
                "Ensure to verify whether these results accurately represent the image and there are no contradictions."
            )
        except Exception as e: 
            error_trace = traceback.format_exc()
            print(colored(f"Vision Agent Could not respond: {str(error_trace)}","red"))
            return f"Vision Agent Is Not Accessibe at the moment\nError:\n{error_trace}"


if __name__ == "__main__":
    async def main():
        data_team=AnalysisTeam(cfg=get_settings(config_path="config.yaml"))
        result=await data_team.forward(query="anaylyze the results", simulation="simulation results is stored at :output/results-12.png,output/results-11.png and output/results-11.csv and output/results-12.csv")
        print(f"the analysis is: {result.to_dict()}")
    import asyncio
    asyncio.run(main())