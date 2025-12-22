import os
import time
import json
import uuid
from os.path import join as ospj
from datetime import datetime, timezone
from epidemiqs.agents.output_structures import EpidemicInfo, SimulationResult, ReflectResult, EpidemicModelInfo, SimulationDetails
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext,Tool, ModelMessagesTypeAdapter
from pydantic_core import to_jsonable_python
from termcolor import colored
from typing import List, Dict
from epidemiqs.agents.tools import CodeExecutor
from epidemiqs.utils.utils import tool_logger
import warnings
from epidemiqs.utils.llm_models import choose_model
from epidemiqs.utils.config import get_settings, Settings
from epidemiqs.utils.utils import log_tokens
from epidemiqs.agents.task_expert_agents import VisionAgent
from epidemiqs.utils.long_term_memory import LTM
from epidemiqs.agents.prompts import system_prompts
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)




#@code_generator.result_validator
def validate_result(result: SimulationDetails) -> bool:
    """
    Validate the result of the code execution.
    
    Args:
        result: The result object containing simulation details.
    
    Returns:
        True if the result is valid, False otherwise.
    """
    print(colored(f"Validating the result of {result}...","green"))
    if not isinstance(result, SimulationDetails):
        raise ValueError("Result is not of type SimulationDetails")

    if not result.simulation_details or not result.stored_result_path:
        raise ValueError("Simulation details or stored result path are empty")
    
    for path in result.stored_result_path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Result path does not exist: {path}. Please correct the output name.")
    
    return result

class SimulationScientist:
    def __init__(self,llm=None,
                 cfg:Settings=get_settings(),
                 phase: str = "Simulation",
                 name:str="SimulationScientist",
                 system_prompt_react: str=system_prompts["simulation_scientist_react"],
                 system_prompt: str=system_prompts["simulation_scientist"],
                 scientist_output_type: dataclass=SimulationDetails,
                 output_type: dataclass=SimulationResult,
                 no_retries: int =None, 
                 end_strategy: str = "exhaustive",
                 repo:str=ospj(os.getcwd(), "output"),
                 ltm_path:str=None
                 ):

        self.name = name
        self.cfg = cfg if cfg is not None else get_settings()
        self.llm = llm if llm is not None else choose_model(self.cfg)["scientists"]

        self.memory = LTM(agent_name=self.name, path=ltm_path).memory if ltm_path else None
        self.conv_length = []
        self.ltm_path = ltm_path or ospj("output","LTM.json")

        self.no_retries = no_retries or self.cfg.workflow.no_retries
        self.tool_retries = 50

        self.react_module = Agent(
            self.llm,
            name=self.name, 
            system_prompt=system_prompt_react,
            output_type=scientist_output_type,
            retries=self.tool_retries,
            output_retries=self.no_retries,
            end_strategy=end_strategy,
            tools=[Tool(self.execute_code, takes_ctx=False), Tool(self.talk_to_vision_expert, takes_ctx=False)],
        )

        self.plan_module = Agent(
            self.llm,
            system_prompt=system_prompt + "\n your job is to exclusively focus on creating a step by step through chain-of-thought to accomplish the task.",
        )

        self.reflect_module = Agent(
            model=self.llm,
            system_prompt=system_prompt + "\n your job is to reflect on the generated response and check if it is accurate (you can check the plot results through vision expert), if not provide accurate instructions in your reflection to correct the answer",
            output_retries=self.no_retries,
            tools=[Tool(self.talk_to_vision_expert, takes_ctx=False)],
            output_type=ReflectResult
        )

        self.output_type = output_type
        self.vision_expert = VisionAgent(cfg=self.cfg)
        self.code_executor = CodeExecutor(self.cfg.workflow.time_out_tools)
        self.phase = phase
        self.repo = repo

        self.input_tokens = 0
        self.output_tokens = 0
        self.tokens = 0
        self.no_tool_exec_iters = 0
        self.no_reflect_iters = 0
        print(colored(f"{phase} initialized with LLMs:\n{self.name}:{self.react_module.model.model_name}\n{self.vision_expert.name}:{self.vision_expert.react_module.model.model_name}", "blue"))
    #run_react 
    async def run_react(self, prompt: str, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                result = await self.react_module.run(prompt, message_history=self.memory)
                self.memory = result.all_messages()

                # Logging
                try:
                    self.log_agent_event("react", prompt, result.output.to_dict())
                except:
                    pass

                print(colored(f"{self.name} result:\n{result.output}\n", "green"))
                self._count_tokens(result)
                return result.output

            except Exception as e:
                if attempt == max_retries - 1:
                    print(colored(f"Max retries reached. Error: {e}", "red"))
                    fallback = SimulationResult(
                        planning="",
                        simulation_details="Simulation failed due to an error.",
                        stored_result_path="",
                        plot_path="",
                        success_of_simulation=False,
                        reasoning_info=str(e),
                    )
                    try:
                        self.log_agent_event("react", prompt, fallback.to_dict(), error=e)
                    except:
                        pass
                    return fallback



    #  token counting
    def _count_tokens(self, message):
        try:
            self.input_tokens += message.usage().input_tokens
            self.output_tokens += message.usage().output_tokens
            self.tokens += message.usage().total_tokens
        except:
            pass


    #    structured logging
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

        log_path = log_path if log_path else ospj(self.repo, f"{self.name}.jsonl")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return log_entry


    #   forward plan/react/reflect 
    async def forward(self, query: str | EpidemicInfo, model: str | EpidemicModelInfo) -> SimulationResult:
        st = time.time()
        is_epidemic_info = isinstance(query, EpidemicInfo)
        is_model_info = isinstance(model, EpidemicModelInfo)
        prompt = (
            f"The description for the initial query is: {query.description if is_epidemic_info else str(query)}.\n"
            +(f"and data path provided by user: {query.data_paths}" if is_epidemic_info and query.data_paths else '')
            +(f"and data regarding network is provided by user: {query.network_path}" if is_epidemic_info and query.network_path else '')
            + "To accomplish the task, the network and mechanistic model are designed as follows:\n"
            + (
                f"network stored at: {model.network_paths}\nmodel details: {model.model_name}\ncompartments: {model.compartments}\ntransitions: {model.transitions}\nparameters: {model.parameters}\ninitial condition description: {model.initial_condition_desc}\ninitial conditions: {model.initial_conditions}\nand logic behind the selected parameters{model.params_reasoning_info}"
                if is_model_info else str(model)
            ) + "\n"
            + "Now, please perform the simulation phase with respect to the provided mechanistic model and network. "
              "Ensure that all scecnarios and models are successfully simulated, there are no missing tasks, and the simulation phase is comprehensive and accurate.\n"
        )

        think_prompt = (
            f"The description for the initial query is: {query.description if is_epidemic_info else str(query)}\n"
            + (f"The general task is: {query.task}.\n" if is_epidemic_info else '')
            +(f"and data path provided by user: {query.data_paths}" if is_epidemic_info and query.data_paths else '')
            +(f"and data regarding network is provided by user: {query.network_path}" if is_epidemic_info and query.network_path else '')
            + "To accomplish the task, the network and mechanistic model are designed as follows:\n"
            + str(model) + "\n"
            + (
                "Now, please create a detailed, step-by-step plan using chain-of-thought reasoning for how to design and set up the simulation phase. "
                "Consider how to use appropriate simulation engines (e.g. we prefer FastGEMF to run the mechanistic model over the static networks).\n"
                "The plan should include:\n"
                "- How to define baseline scenarios for comparison\n"
                "- How to incorporate variations to test different conditions\n"
                "- How to ensure the simulations provide meaningful insights into the modeled system\n"
                "- How to verify that all simulations are executed successfully\n"
                "Focus exclusively on scenario design and simulation setup to accomplish **all** the requested tasks.\n"
            )
        )

        #  PLAN 
        if self.cfg.workflow.scientist_modules.plan:
            try:
                plan = await self.plan_module.run(think_prompt)
                print(colored(f"Plan:\n{plan.output}", "blue"))
                self.log_agent_event("plan", think_prompt, plan.output)
                self._count_tokens(plan)

                result = await self.run_react(
                    prompt + f"Here is the suggested plan to follow:\n{plan.output}\n"
                )

            except Exception as e:
                print(colored(f"Error in plan step: {e}", "red"))
                self.log_agent_event("plan", think_prompt, error=str(e))
                plan = None
                result = await self.run_react(prompt)

        else:
            result = await self.run_react(prompt)
            plan = None

        print(colored(f"Results:\n {result}\n","green"))
        print(f"\n {'='*10} \n")

        #  REFLECTION LOOP 
        for _ in range(self.cfg.workflow.reflection_max_iters):
            if self.cfg.workflow.scientist_modules.reflect:

                reflection_prompt = (
                    f"""Now I want you to reflect using chain-of-thought on the simulation generated results:\n{result}\n to check they are done correctly and successfully acording to the initial query:\n{prompt}\n. 
                    Are all the models simulated successfully? is Simulation phase comprehensive and accurate? 
                    are there any mistakes or inaccuracies in the code or any drift in the simulation results? 
                    are all planned scenarios simulated correctly?
                    you have vision expert to help you analyze the plot to ensure the success of the simulation.
                    check the saved plots to ensure they are accurate and represent the simulation results correctly. 
                    If you find any deviation from the expected results, provide instructions to correct it. 
                    If satisfied, set revise_needed=False."""
                )

                try:
                    reflection = await self.reflect_module.run(
                        reflection_prompt,
                        message_history=self.memory
                    )
                    self.log_agent_event("reflection", reflection_prompt, reflection.output.to_dict())
                    self._count_tokens(reflection)

                except Exception as e:
                    print(colored(f"Reflection error: {e}", "red"))
                    self.log_agent_event("reflection", reflection_prompt, error=str(e))
                    reflection = ReflectResult(reflection="", revise_needed=False)

                if reflection.output.revise_needed:
                    self.no_reflect_iters += 1
                    print(colored(f"Revision Needed, Reflection:\n {reflection.output.reflection} \n", "yellow"))
                    revise_prompt = f"Please consider the reflection:\n {reflection.output.reflection}\n to accomplish the task accurately "
                    result = await self.run_react(revise_prompt)

                    if self.memory and len(self.memory) > 0:
                        self.conv_length.append(len(self.memory))
                    self._trim_context_memory()

                else:
                    print(colored(f"No revision needed!Reflection:\n {reflection.output.reflection}", "green"))
                    break

        #  COPILOT 
        if self.cfg.workflow.copilot:
            user_feedback = "no"
            while user_feedback.lower() not in {'yes','y'}:
                input_text = colored(
                    f"\nThe generated results are as:\n"
                    + "-"*25
                    + f"{result}\n"
                    + "-"*25
                    + "If you have any comments provide them. If satisfied enter \"yes\"\n",
                    "white"
                )
                user_feedback = input(input_text)

                if user_feedback.lower() not in {"yes", "y"}:
                    revise_prompt = f"Now based on the user feedback\n {user_feedback}, address the concerns by incorporating the feedback for simulations\n"
                    print(colored(f"Revised Results:\n", "green"))
                    result = await self.run_react(revise_prompt)

        #  SAVE MEMORY & TOKENS 
        try:
            self.total_tokens = self.input_tokens + self.output_tokens
            LTM.save_LTM_direct(agent_name=self.name, agent_memory=self.memory, path=self.ltm_path)
            log_tokens(repo=self.repo,csv_name="tokens_by_phase.csv", 
                   agent_name=self.phase, 
                   llm_model=self.react_module.model.model_name,
                   input_tokens=self.input_tokens,
                   output_tokens=self.output_tokens, total_tokens=self.tokens,time=time.time()-st)
            log_tokens(
                repo=self.repo,
                agent_name=self.name,
                llm_model=self.react_module.model.model_name,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                total_tokens=self.tokens
            )

        except Exception as e:
            print(colored(f"Error saving LTM/tokens: {e}", "red"))


        return self.output_type(
            planning=plan.output if plan else "No planning done",
            simulation_details=result.simulation_details,
            stored_result_path=result.stored_result_path,
            plot_path=result.plot_path,
            success_of_simulation=result.success_of_simulation,
            reasoning_info=result.reasoning_info
        )


    async def talk_to_vision_expert(self, query: str, image_paths:List[str]) -> str:
        print(colored(f"\nTalking to vision agent...\n inquiry:{query} \n at path:{image_paths}", "blue"))
        try:
            resp = await self.vision_expert.forward(query=query, image_paths=image_paths)
            print(colored(f"Vision Agent Response:{resp}", "white"))
            return resp
        except Exception as e:
            print(colored(f"Vision Agent Error:\n {str(e)}\n", "red"))
            return "Vision Expert Is Not Accessible at the moment"


    #   EXECUTE CODE 
    async def execute_code(self, code: str, return_vars: List[str] = None, script_name: str="simulation.py") -> str:
        self.no_tool_exec_iters += 1
        result = await self.code_executor.execute(code, return_vars, script_name=script_name, write_to_file=True)
        if self.no_tool_exec_iters > self.tool_retries-5:
            message=f"\nNote: You have executed more than {self.no_tool_exec_iters} code blocks, You have {self.tool_retries - self.no_tool_exec_iters} more tries left before reaching the maximum limit of {self.tool_retries} code executions. Make sure to finalize your answer ASAP."    
            result+=message   
            print(colored(f"Code executed successfully. Result:\n{result}\n", "green"))
        return result


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
            
            
if __name__ == "__main__":

    async def run_simulation() -> SimulationResult:        
        cfg=get_settings("config.yaml")
        code_agent=SimulationScientist(cfg=cfg,ltm_path=None)
        simulation= await code_agent.forward(query="please run SIR with R0=4,5,6,7 on ER with 100 node and average degree 4, 100 simulation, analyze the results using sklearn and MINMAX VARIATION of results" ,model=None)
        return simulation
    import asyncio
    result=asyncio.run(run_simulation())



    #print(colored(f"fastgemf version: {fg.__version__}","green")    )
    print(colored(f"Simulation Result: {result}","white"))