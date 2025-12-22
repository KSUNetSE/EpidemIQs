import os
import json
import traceback
import time
from os.path import join as ospj
from epidemiqs.utils.utils import log_tokens
from epidemiqs.utils.utils import move_to_destination
from epidemiqs.agents.report_generation import generate_report_sections
from epidemiqs.utils.config import get_settings,reload_settings
from epidemiqs.agents.prompts import system_prompts
from pydantic_ai import Agent
from pydantic_ai.tools import Tool
from termcolor import colored, cprint
from time import sleep
from epidemiqs.utils.llm_models import choose_model
from epidemiqs.agents.discovery import DiscoveryScientist, EpidemicInfo
from epidemiqs.agents.modeling import ModelingTeam,EpidemicModelInfo
from epidemiqs.agents.data_analysis  import AnalysisTeam
from epidemiqs.agents.simulation  import SimulationScientist,SimulationResult
from epidemiqs.agents.task_expert_agents import SecretaryAgent, SecretaryAgentResult
from epidemiqs.utils.move_files import move_to_destination, create_experiment_folder
from epidemiqs.utils.config import Settings, get_settings
from epidemiqs.utils.reference_validator import run_ref_check
from epidemiqs.agents.output_structures import AnalysisResult
reload_settings()

class Epidemiqs():
    def __init__(self,query: str=None,cfg:Settings=None):     
        self.cfg= cfg or get_settings()
        self.user_prompt = query or self.cfg.query
        self.discovery_scientist=DiscoveryScientist(cfg=self.cfg)
        self.modeler_scientist=ModelingTeam(cfg=self.cfg)#model_name=model_name)
        self.simulator_scientist=SimulationScientist(cfg=self.cfg)#_model_name=model_name)
        self.data_scientist=AnalysisTeam(cfg=self.cfg)#model_name=model_name)
        self.supervisor_queue = asyncio.Queue() 
        self.repo = ospj("output","log.json")
        self.iteration = 0
        self.tokens=0
        self.secretary_agent=SecretaryAgent()
        self.user_output_dir=self.cfg.paths.output_dir
    


    def load_log(self):
        """Load existing log file if available."""
        if os.path.exists(self.repo):
            with open(self.repo, "r") as file:
                self.log_data = json.load(file)
        else:
            self.log_data = {}

    def save_log(self):
        """save the log data to a JSON file."""
        with open(self.repo, "w") as file:
            json.dump(self.log_data, file, indent=4)

    def log_result(self, agent_name:str, input_data, output_data)-> None:
        """Log each Phase's input and output to the JSON file."""
        iteration_key = f"iteration_{self.iteration}"
        
        if agent_name not in self.log_data:
            print(f"Creating log entry for agent: {agent_name}")
            self.log_data[agent_name] = {}

        if iteration_key not in self.log_data[agent_name]:
            self.log_data[agent_name ][iteration_key] = []

        self.log_data[agent_name ][iteration_key].append({
            "data": output_data
        })

        self.save_log()


    async def _sequential_workflow(self):
        print(self.cfg.paths)
        data_file_name=os.path.basename(self.cfg.paths.data_path)
        move_to_destination(self.cfg.paths.data_path, ospj(os.getcwd(),"output"),move=False,ignore_names=[])
        move_to_destination(ospj(os.getcwd(),"output"),ospj(os.getcwd(),"output","deleted_files"),ignore_names=[data_file_name])
        self.load_log()
        global iteration 
        iteration = 0
        continue_loop = True
        self.iteration = 0
        workflow_trigger = False
        
        if not self.user_prompt:
            self.ask_user()
        try:
            secretary_response = await self.secretary_agent.run(self.user_prompt)
        except Exception as e:
            print(colored(f"Error in secretary agent: {str(e)}\n","red"))
            
            secretary_response = SecretaryAgentResult(
                in_scope_of_EpidemIQs=False)
            if input(colored("There was an error in classifying your query. if you are sure you want your query to be processed with EpidemIQs, please enter \"yes\" to continue or \"no\" to abort: \n","cyan")).lower() in ["yes","y"]:
                workflow_trigger = True
                #secretary_response=
            else:
                workflow_trigger = False
                print(colored("Aborting the workflow, please try again later!","red"))
                
        if secretary_response.in_scope_of_EpidemIQs or workflow_trigger:
            discovery_scientist_results = None
            st=time.time()
            timing_data = {}
            
            start_time = time.time()
            try:
                discovery_scientist_results = await self.discovery_scientist.forward(
                    f"{self.user_prompt}"
                )
                self.log_result("Discovery", [self.user_prompt], discovery_scientist_results.to_dict() )
            except Exception as e:
                print(colored(f"Error in Discovery Scientist agent: {str(e)}","red"))
                error_trace=traceback.format_exc()
                discovery_scientist_results = {"error in Discovery Phase": str(error_trace)}
                self.log_result("Discovery", [self.user_prompt], discovery_scientist_results)
                
            timing_data["Discovery"] = time.time() - start_time
            self.log_result("USER_QUERY", "", str(self.user_prompt))
            self.tokens+=self.discovery_scientist.tokens if discovery_scientist_results else 0
            print(colored(25*'-',"white")   )
            print(colored(f'Discovery Done. Next Phase: Modeling',"white"))
            print(colored(25*'-',"white")   )
    
            sleep(1)
            while continue_loop:                 
                start_time = time.time()
                
                try:
                    epidemic_modeler_result = await self.modeler_scientist.forward(query=discovery_scientist_results)
                    self.log_result("Modeling", "", epidemic_modeler_result.to_dict() )

                except Exception as e:
                    error_trace=traceback.format_exc()
                    print(colored(f"Error in Epidemic Modeler agent: {str(e)}","red"))
                    epidemic_modeler_result = {"error in Modeling Phase": str(error_trace)}
                    self.log_result("Modeling", "", epidemic_modeler_result )
                    
                timing_data["Modeling"] = time.time() - start_time
                self.tokens+=self.modeler_scientist.tokens if epidemic_modeler_result else 0
                print(colored(25*'-',"white")   )
                print(colored(f'Modeling Done. Next Phase: Simulation',"white"))
                print(colored(25*'-',"white")   )
                sleep(1)      
                        
                start_time = time.time()
                try:    
                    simulator_agent_result = await self.simulator_scientist.forward(query=discovery_scientist_results, model=epidemic_modeler_result)
                    self.log_result("Simulation", "prompt", simulator_agent_result.to_dict() )

                except Exception as e:
                    error_trace=traceback.format_exc()
                    print(colored(f"Error in Simulator Agent: {str(e)}","red"))
                    simulator_agent_result = {"error in Simulation Phase": str(error_trace)}
                    self.log_result("Simulation", "prompt", simulator_agent_result)
                #timing_data["simulator_agent"] = time.time() - start_time
                self.tokens+=self.simulator_scientist.tokens if simulator_agent_result else 0
        
                print(colored(25*'-',"white")   )
                print(colored(f'Simulation Done. Next Phase: Analysis',"white"))  
                print(colored(25*'-',"white")   )
                sleep(1)              
                start_time = time.time()
                try:    
                    analyzer_agent_result = await self.data_scientist.forward(query=discovery_scientist_results, simulation=simulator_agent_result)
                    self.log_result("DataAnalysis", "", analyzer_agent_result.to_dict() )

                except Exception as e:
                    error_trace=traceback.format_exc()
                    print(colored(f"Error in Analyzer Agent: {str(e)}","red"))
                    analyzer_agent_result = {"error in Analysis Phase": str(error_trace)}
                    self.log_result("DataAnalysis", "", analyzer_agent_result)


                timing_data["analyzer_agent"] = time.time() - start_time
                self.tokens+=self.data_scientist.tokens if analyzer_agent_result else 0
                
                print(colored(25*'-',"white")   )
                print(colored(f'Analysis Done. Next Phase: Reporting',"white"))
                print(colored(25*'-',"white")   )   
                
                sleep(1) 
                self.iteration += 1
                continue_loop=False

            start_time = time.time()
            try:
                await generate_report_sections(log_name="log.json", lit_review_name="literature_review.json")
            except Exception as e:
                print(colored(f"Error generating report sections: {e}", "red"))
            timing_data["report_generation"] = time.time() - start_time
            log_tokens(repo=ospj(os.getcwd(), "output"),csv_name="tokens_by_phase.csv",agent_name="Total",
                        llm_model=choose_model(self.cfg)["scientists"],total_tokens=self.tokens,time=time.time()-st)
            # Write timing data to file
            with open(ospj("output","phase_timing.txt"), "w") as timing_file:
                timing_file.write("Agent Execution Timing Report\n")
                timing_file.write("===========================\n\n")
                for agent, execution_time in timing_data.items():
                    timing_file.write(f"{agent}: {execution_time:.2f} seconds\n")
                    timing_file.write(f"\nTotal workflow time: {sum(timing_data.values()):.2f} seconds\n")
            try:
                run_ref_check(
                    tex_path=ospj("output","initial_report.tex"),
                    json1_path=ospj("output","literature_review.json"),
                    json2_path=ospj("output","back_grounds.json"),
                    threshold=75.0,
                    out_json=None,
                    out_excel="hallucination_reference_summary.xlsx",
                    model_name="all-MiniLM-L6-v2")
            except Exception as e:  
                print(colored(f"Error running reference validator: {e}", "red"))    
            user_output_path=create_experiment_folder(self.user_output_dir,self.cfg.name)
            move_to_destination(ospj(os.getcwd(),"output"), user_output_path)
            
    async  def _dynamic_workflow(self):
        pass    # Not implemented yet, will be added in future versions
    
    async def run(self,key:str="sequential"):
        #self.cfg=self.cfg
        #self.user_prompt = query or self.user_prompt
        await self._sequential_workflow()



    @classmethod
    def ask_user(self):
        self.user_prompt = input("explain your epidemic scenario: ")   


import asyncio
import time
if __name__ == "__main__":
    async def main(prompt=None): 
        sample_agent=Epidemiqs(query='run sir over 100 nodes ER, beta=0.3, gamma=0.1 for 160 days, no online search, no user, just go agent',cfg=get_settings())
        await sample_agent.run()

    start_time = time.time()
    asyncio.run(main())
    elapsed_time = time.time() - start_time
    os.system(f"Say 'The simulation is completed in {elapsed_time:.2f} seconds  '")
    
