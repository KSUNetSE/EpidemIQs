from typing import Any, Callable, Generic, Optional, Type, TypeVar, List
import os
import time
import traceback
import asyncio
from termcolor import colored
from pydantic.dataclasses import dataclass
from epidemiqs.agents.tools import CodeExecutor
from pydantic_ai import BinaryContent, Agent, Tool
from epidemiqs.config import Settings, get_settings
from epidemiqs.utils.llm_models import choose_model
from epidemiqs.utils.utils import log_tokens
import json
import uuid
from datetime import datetime
import os
from os.path import join as ospj

@dataclass
class ReflectResult:
    revise_needed: bool
    reflection_notes: str
    def to_dict(self):  
        return {
            "revise_needed": self.revise_needed,
            "reflection_notes": self.reflection_notes
        }   
@dataclass
class Result:
    answer: str
    #reasoning_info: str
    def to_dict(self):
        return {
            "answer": self.answer#,
            #"reasoning_info": self.reasoning_info
        }


TPhaseOutput = TypeVar("TPhaseOutput")        # Final structured output (e.g., EpidemicInfo)
TScientistOutput = TypeVar("TScientistOutput")  # Raw agent output type (e.g., DiscoveryScientistInfo)


class BaseScientist(Generic[TPhaseOutput, TScientistOutput]):
    """
    Generic scientist agent architecture.

    This class encapsulates the common pattern:
      - Optional plan module (think/plan)
      - Main react module (tools + reasoning)
      - Optional reflect module (self-critique + revision)
      - Optional copilot loop with the user
      - Optional long-term memory
      - Optional math / retrieval / logging, injected via tools and callbacks

    You customize behavior via constructor arguments instead of changing this class.
    """

    def __init__(
        self,
        # Core
        llm: Any,
        cfg: Settings,
        name: str,
        phase: str,
        system_prompt: str,
        scientist_output_type: Type[TScientistOutput],
        phase_output_type: Type[TPhaseOutput],
        tools: List[Any],
        default_tool:bool =True,

        # Optional modules
        enable_plan: bool = True,
        enable_reflect: bool = True,
        enable_copilot: bool = False,
        plan_prompt_suffix: str = (
            "Now think step by step and plan how to address the query as best as possible. "
        ),
        reflect_prompt_suffix: str = (
            "Now reflect on the generated response and check accuracy, completeness, "
            "and deviation from the user request. If revision is needed, provide "
            "step-by-step instructions to correct the answer. Otherwise set revise_needed=False."
        ),

        # Memory / logging / paths
        repo: str = os.path.join(os.getcwd(), "output"),
        ltm_path: Optional[str] = None,
        ltm_cls: Optional[Any] = None,          # e.g., LTM class
        log_tokens_fn: Optional[Callable[..., None]] = None,  # e.g., log_tokens

        # Agent factory (to decouple from concrete Agent class if needed)
        agent_cls: Any = None,                  # e.g., Agent
        react_end_strategy: str = "exhaustive",
        model_settings: Optional[dict] = None,

        # Output builder: how to convert raw scientist_output -> phase_output
        output_builder: Optional[
            Callable[[TScientistOutput, "BaseScientist"], TPhaseOutput]
        ] = None,
    ):
        self.cfg = cfg
        self.llm = llm
        self.name = name
        self.phase = phase
        self.system_prompt = system_prompt
        self.phase_output_type = phase_output_type
        self.scientist_output_type = scientist_output_type
        self.repo = repo
        self.enable_plan = enable_plan
        self.enable_reflect = enable_reflect
        self.enable_copilot = enable_copilot
        self.log_tokens_fn = log_tokens_fn
        self.output_builder = output_builder
        self.input_tokens = 0
        self.output_tokens = 0
        self.no_reflect_iters = 0
        self.code_executor = CodeExecutor()
        self.results = {}
        self.code_exec_iter=0
        self.retries=0
        self.tools= [Tool(self.execute_code, takes_ctx=False), Tool(self.retrieve_images, takes_ctx=False), Tool(self.scan_for_formats, takes_ctx=False)]+ tools if default_tool else tools
                     #Tool(self.dummy_tool, takes_ctx=False)] 
                     
        if not os.path.exists(self.repo):
            os.makedirs(self.repo)

        self.ltm_path = ltm_path or os.path.join(self.repo, f"{self.name}_LTM.json")
        self.memory = None
        self.conv_length: List[int] = []

        if ltm_cls is not None:
            self.memory = ltm_cls(agent_name=self.name, path=self.ltm_path).memory

        model_settings = model_settings or {}
        #model_settings.setdefault("parallel_tool_calls", False)

        if agent_cls is None:
            raise ValueError("agent_cls (Agent class) must be provided to BaseScientist")

        # Main (ReAct) module
        self.react_module = agent_cls(
            model=self.llm,
            name=self.name,
            system_prompt=self.system_prompt,
            output_type=self.scientist_output_type,
            end_strategy=react_end_strategy,
            #model_settings=model_settings,
            output_retries=self.cfg.workflow.no_retries,
            retries=int(200),
            tools=self.tools,
        )

        # Optional plan module
        self.plan_module = None
        if self.enable_plan:
            self.plan_module = agent_cls(
                name=f"{self.name}_PlanModule",
                model=self.llm,
                system_prompt=self.system_prompt + "\n" + plan_prompt_suffix,
            )

        # Optional reflect module
        self.reflect_module = None
        if self.enable_reflect:
            self.reflect_module = agent_cls(
                name=f"{self.name}_ReflectModule",
                model=self.llm,
                system_prompt=self.system_prompt + "\n" + reflect_prompt_suffix,
                output_type=ReflectResult,  # can be overridden
            )

        print(colored(
            f"{self.phase} scientist initialized:\n"
            f"  Name: {self.name}\n"
            f"  Model: {self.react_module.model.model_name}\n",
            "light_blue"
        ))

    def _get_reflect_output_type(self) -> Any:
        """
        Override in subclass if you want a concrete reflection output schema.
        Default: use scientist_output_type (no special schema).
        """
        return self.scientist_output_type

    async def forward(self, query: str|List) -> TPhaseOutput:
        """
        Generic forward method:
          1) [Optional] planning
          2) Main reaction (tools + reasoning)
          3) [Optional] reflection loop
          4) [Optional] copilot interaction with user
          5) [Optional] logging and memory saving
          Comments: The benchmarks show significance improvements with planning and reflection (please check the EpidemiQs paper or the DSBench benchmark results).
        """
        start_time = time.time()
        base_prompt = (
            f"User query:\n{query}\n\n"
            "You must gather all relevant information using your tools and knowledge "
            "to answer the query to the best of your ability.\n"
        )
        prompt = base_prompt

        if self.enable_plan and self.plan_module is not None:
            try:
                think_prompt = prompt + (
                    "\nNow create a detailed step-by-step plan "
                    "for how you will  address the question."
                )
                plan_result = await self.plan_module.run(
                    think_prompt,
                    message_history=self.memory
                )
                self.log_agent_event("plan", think_prompt, plan_result.output)
                self.input_tokens += plan_result.usage().input_tokens 
                self.output_tokens += plan_result.usage().output_tokens
                prompt += f"\nSuggested plan:\n{plan_result.output}\n"
                print(colored(f"Plan:\n{plan_result.output}\n", "blue"))
            except Exception as e:
                error_trace = traceback.format_exc()
                print(colored(
                    f"Error in {self.name} planning step: {e}\n{error_trace}",
                    "red"
                ))

        try:
            if query and isinstance(query, str):
                result = await self.run_react(prompt)
            elif query and isinstance(query, list):
                result = await self.run_react(query)        


            if self.enable_reflect and self.reflect_module is not None:
                for _ in range(self.cfg.workflow.reflection_max_iters):
                    ref_prompt = (
                        "Initial user query:\n"
                        f"{query}\n\n"
                        "Final Generated response to query is:\n"
                        f"{result.answer}\n\n"
                        "Now reflect on this response to check for accuracy and possible mistakes. If it is not accurate, provide step-by-step instructions on how to revise it.\n"
                    )
                    
                    reflection = await self.reflect_module.run(
                        ref_prompt,
                        message_history=self.memory
                    )
                    self.log_agent_event("reflect", ref_prompt, reflection.output.to_dict())
                    self.input_tokens += reflection.usage().input_tokens
                    self.output_tokens += reflection.usage().output_tokens
                    print(colored(f"Reflection:\n{reflection.output}\n", "yellow"))
                    self.no_reflect_iters += 1
                    revise_needed = getattr(reflection.output, "revise_needed", False)
                    if not revise_needed:
                        break

                    prompt += (
                        f"\nPlease consider the following reflection:\n"
                        f"{reflection.output}\n"
                        "and provide a more accurate, precise answer accordingly.\n"
                    )
                    result = await self.run_react(
                        prompt                        
                    )
                    print(colored(
                        f"{self.name} revised result:\n{result}\n",
                        "green"
                    ))
                    self.conv_length.append(len(self.memory) if self.memory else 1)
                    self._trim_context_memory()

            if self.enable_copilot:
                user_input = "no"
                while user_input.lower() not in ["yes", "y"]:
                    input_text = colored(
                        f"Current result:\n{result.output}\n\n"
                        "If the output is good, type \"yes\".\n"
                        "Otherwise, type your comments:\n",
                        "white"
                    )
                    user_input = await asyncio.to_thread(input, input_text)
                    if user_input.lower() not in ["yes", "y"]:
                        prompt = (
                            f"Consider the following user comments:\n{user_input}\n"
                            "Provide a more accurate and precise answer accordingly.\n"
                        )
                        result = await self.run_react(
                            prompt,
                            
                        )
                        print(colored(
                            f"{self.name} revised after user comments:\n"
                            f"{result}\n",
                            "green"
                        ))


            elapsed = time.time() - start_time
            if self.log_tokens_fn is not None:
                try:
                    self.log_tokens_fn(
                        repo=self.repo,
                        agent_name=self.name,
                        llm_model=self.react_module.model.model_name,
                        input_tokens=self.input_tokens,
                        output_tokens=self.output_tokens,
                        total_tokens=self.input_tokens + self.output_tokens,
                        time=elapsed,
                    )
                    self.log_tokens_fn(
                        repo=self.repo,
                        csv_name="tokens_by_phase.csv",
                        agent_name=self.phase,
                        llm_model=self.react_module.model.model_name,
                        input_tokens=self.input_tokens,
                        output_tokens=self.output_tokens,
                        total_tokens=self.input_tokens + self.output_tokens,
                        time=elapsed,
                    )
                except Exception as e:
                    print(colored(f"Error logging tokens: {e}", "red"))

            if hasattr(self, "LTM") and self.memory is not None:
                #  if you want to use a static LTM API
                try:
                    self.LTM.save_LTM_direct(
                        agent_name=self.name,
                        agent_memory=self.memory,
                        path=self.ltm_path,
                    )
                except Exception as e:
                    print(colored(f"Error saving LTM: {e}", "red"))

            raw_output: TScientistOutput = result

            if self.output_builder is not None:
                return self.output_builder(raw_output, self)

            if hasattr(raw_output, "to_dict"):
                return self.phase_output_type(**raw_output.to_dict())

            #fallback: just wrap as-is
            #return self.phase_output_type(raw_output)
            return result

        except Exception as e:
            error_trace = traceback.format_exc()
            print(colored(f"Error in {self.name}: {e}\n{error_trace}", "red"))
            # You can also choose to raise instead of returning a string
            return self.phase_output_type(
                **{"error": f"Error in {self.name}: {e}\n{error_trace}"}
            )

    async def run_react(self,prompt: str,max_retries: int=3) -> str:
        for attempt in range(max_retries):
            try:
                result = await self.react_module.run(prompt, message_history=self.memory)
                self.memory = result.all_messages()
                self.log_agent_event("react", prompt, result.output.to_dict())
                print(colored(f"{self.name} result:\n{result.output}\n", "green"))
                self.input_tokens += result.usage().input_tokens 
                self.output_tokens += result.usage().output_tokens
                return result.output  # exit the function with the result

            except Exception as e:
                if attempt == max_retries - 1:
                    message=Result(answer=f"Agent failed after maximum retries {max_retries} due to error: {e}")
                    self.log_agent_event("react", prompt, message, error=e)
                    
                    return message
                
                
    def log_agent_event(self,
        module_name: str,
        input_data=None,
        output_data=None,
        error=None,
        meta: dict | None = None,
        log_path: str="Scientist_agent_log.jsonl"
    ):
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
        
        input_data = input_data if isinstance(input_data, str) else "[non-string input]"
        log_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "module": module_name,
            "input": input_data,
            "output": output_data,
            "error": str(error) if error else None,
            "no_exec_iter": self.code_exec_iter,   
            "no_reflections":self.no_reflect_iters, 
            "meta": meta or {}
        }

        # append as a JSON line
        log_path=self.name + "_" + module_name + "_log.jsonl"
        with open(ospj(self.repo, log_path), "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False))
            f.write("\n")

        return log_entry
    
    def _trim_context_memory(self) -> None:
        """
        Trim context memory (action_{t-2},result_{t-2}) reflection_{t-2} ) 
        """
        
        try:
            if len(self.conv_length) > 1 and self.memory:
                print(colored("\nTrimming memory...\n", "yellow"))
                n = self.conv_length[0]
                self.memory = [self.memory[0]] + self.memory[n:]
                self.conv_length = [len(self.memory)]
        except Exception as e:
            print(colored(f"Error trimming memory: {e}", "red"))

    async def execute_code(self, code: str, return_vars: List[str] = None,  script_name: str="choose-proper-name.py") -> str:            
        """
        Execute Python code in a persistent environment. (never print in the code, return variables instead)
        
        Args:
            code: The Python code to execute.
            return_vars: List of variable names to return after execution.
            script_name: specify the name of the script file to write code to (choose according to the content of your code). Do not use default name!
        
        Returns:
            string containing execution success message and paths and returned variables.
        """
        self.code_exec_iter+=1
        
        result = await self.code_executor.execute(code, return_vars, script_name=script_name,write_to_file=True)
        if self.code_exec_iter >45:
            result+=f"  Note: You have executed more than 45 code blocks, You have {50 - self.code_exec_iter} more tries left before reaching the maximum limit of 50 code executions. Make sure to finalize your answer soon."
        print(colored(f"Code execution  Result:\n{result}\n", "green"))
        return result
    
    def _reset(self) -> None:
        """
        Reset internal state for a new query.
        """
        self.code_executor.code_history={"":""}
        #self.image = image
        #self.excel = excel
        self.memory = None
        self.conv_length = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.code_exec_iter=0
        self.no_reflect_iters = 0
    
    def retrieve_images(self, image_paths: List[str]) -> str:
        """Retrieve insights from images based on user query."""
        print(colored(f"Retrieving images: {image_paths}", "green"))
        
        prompt = [""]
        supported_formats = (".png", ".jpg", ".jpeg")

        for path in image_paths:
            file_name=os.path.basename(path) 
            try:
                if not os.path.exists(path):
                    prompt.append(f"{file_name} not found at {path}!!! Make sure the file exists and is accessible And Ensure to mention in your response that this file is not available!!\n")
                    continue  # Skip this file and move to the next one 
                
                if not file_name.lower().endswith(supported_formats):
                    prompt.append(f"{file_name} is not in PNG format!!! Only PNG format is supported. "
                                f"Ensure to mention in your response that this file format is not supported!!\n")
                    continue  # Skip this file and move to the next one
                
                with open(path, "rb") as image_file:
                    image_data = image_file.read()
                file_name = os.path.basename(path)
                
                prompt.append(f"Image named as:'{file_name}'loaded successfully\n")
                prompt.append(BinaryContent(data=image_data, media_type='image/png'))
                
                
            except Exception as e:
                prompt.append(f"Error reading {path}: {e}")
        #full_prompt="\n".join(prompt)
        return prompt
    def scan_for_formats(self, format: str) -> list[str]:
        """
        Recursively scan the repository for a given file extension, for possible mistakes in provided file names.
        
        Parameters
        ----------
        format : str
            File extension to search for (e.g., '.json', '.yaml', '.md').

        Returns
        -------
        list[str]
            A list of full paths to matching files.
        """
        print(colored(f"Scanning for files with format: {format}", "cyan"))
        if not format.startswith("."):
            format = "." + format

        ignore_dirs = {"log", "token", "tokens_by_phase"}
        matched_files = []

        for root, dirs, files in os.walk(self.repo):
            # Remove ignored directories before continuing the walk
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            for f in files:
                if f.endswith(format):
                    matched_files.append(os.path.join(root, f))
        print(colored(f"Found files: {matched_files}", "cyan"))
        return matched_files
    
    
    def dummy_tool(self, key:bool=True) -> str:
        """
        A dummy tool that does nothing.
        """
        self.code_exec_iter+=1
        print(colored(f"Dummy tool called with key={self.code_exec_iter}", "green"))
        pass
    
    
if __name__ == "__main__":  
    cfg = get_settings()
    discovery_scientist = BaseScientist[
        Result,
        Result
    ](
        llm=choose_model(cfg)["scientists"],
        cfg=cfg,
        name="DiscoveryScientist",
        phase="Discovery",
        #system_prompt="  You are a highly intelligent Data Scientist who excels at data analysis and give precise answer to the questions based on the data provided. after providing your answer, always give a concise reasoning about how you reached to this answer( for future improvements, user does not see this reasoning).",
        system_prompt="you listen to what ever I say",
        scientist_output_type=Result,
        phase_output_type=Result,

        enable_plan=cfg.workflow.scientist_modules.plan,
        enable_reflect=cfg.workflow.scientist_modules.reflect,
        enable_copilot=cfg.workflow.copilot,
        repo=os.path.join(os.getcwd(), "output"),
        ltm_path=os.path.join("output", "LTM.json"),
        ltm_cls=None,
        tools=[],
        log_tokens_fn=log_tokens,
        agent_cls=Agent,
    )
    print(colored(f"discovery_scientist initialized: {discovery_scientist.cfg.workflow.reflection_max_iters}.", "light_blue"))
    print("Scientist agent initialized.")
    print(f"Model: {discovery_scientist.react_module.model.model_name}")
    result=asyncio.run(discovery_scientist.forward("call dummy tool until it returns 200. It WILL RETURN 200, just keep calling."))
    print("Final result:", result)
    print(f"Total input tokens: {discovery_scientist.input_tokens}"
        f", output tokens: {discovery_scientist.output_tokens}"
        f", total tokens: {discovery_scientist.input_tokens + discovery_scientist.output_tokens}")
    agent=Agent()   