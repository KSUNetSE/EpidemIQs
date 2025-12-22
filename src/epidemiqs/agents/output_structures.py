# data strcuture for output
from dataclasses import dataclass
import os,json
from typing import Optional, Dict,List
from dataclasses import field



@dataclass
class DiscoveryScientistInfo:
    """
    Dataclass to hold epidemic information.
    """
    description: str
    task: str #Literal["simulation","mitigaion"]   
    goal: str    # e.g., "target is MSE<0.1", "target is accuracy>0.9", etc.
    pathogen: str  # e.g., "COVID-19", "Ebola"
    compartment_model: str  # SI, SIR, SEIR, etc.
    disease_type: str  # "STI", "vector-borne", "zoonotic", etc.
    current_condition: str  
    R_0: Optional[float] = None  # If we know R_0, it can be very helpful
    network_path: Optional[str] = None  # User must provide either a path or a description
    contact_network_structure: Optional[str] = None  # if path is not provided, YOU MUST suggest a network structure
    math_solution: Optional[str] = None  # If a mathematical solution is availappropriate, provide it here
    data_paths: Optional[Dict[str,str]] = None  # Path where data is stored (if provided)
    reasoning_info: str= None  # Path where reasoning information is stored (if provided)


    def to_dict(self,iteration=0):
        return self.__dict__.copy()

    def to_json(self, iteration=0):
        """Load existing JSON file if available and append the new iteration data before saving."""
        cwd = os.getcwd()
        path = os.path.join(cwd, "output", f"{DiscoveryScientist}_info.json")  
        if os.path.exists(path):
            with open(path, "r") as f:
              try:
                  existing_data = json.load(f)
              except json.JSONDecodeError:
                  existing_data = {} 
        else:
            existing_data = {}
        data_dict = self.to_dict(iteration)
        existing_data[f"iteration_{iteration}"] = data_dict

        # Save updated data
        with open(path, "w") as f:
            json.dump(existing_data, f, indent=4)
@dataclass
class EpidemicInfo(DiscoveryScientistInfo):
    mathematic_full_response:str= ""  # Full response from the mathematician agent
    def to_dict(self, iteration=0):
        return self.__dict__.copy()
@dataclass
class ReflectResult:
    """Class to hold search results."""
    reflection: str
    revise_needed: bool

    def to_dict(self):
        return self.__dict__.copy() 
    
@dataclass
class ReflectResult:
    """Class to hold search results."""
    reflection: str
    revise_needed: bool

    def to_dict(self):
        return self.__dict__.copy() 

 # Structured Output Classes
@dataclass
class MechanisticModel:
    """

    Model for a mechanistic model over network.
    Attributes:
        name: str # e.g., SIR, SIRV, SEIRH
        compartments: list[str], # ["S", "I","R"] 
        transitions: Dct[str: str] # {"from state A to state B": "rate"}
        reasoning_info: str
    """    
    name: str # e.g., SIR, SIRV, SEIRH
    compartments: List[str] # e.g., ["S", "I", "R"]
    transitions: Dict[str, str]  # {"S -(I)-> I": "beta", "I -> R": "gamma"} S-(I)->I, (I) is inducer state
    reasoning_info: str
    def to_dict(self):  
        return self.__dict__.copy()  # Return a copy of the dictionary representation of the instance

@dataclass
class NetworkStructure:
    """
    netowrk_details:str # explaining the network structure
    network_paths: List[str] # where the  network(s) saved.
    plot_paths: Dict[str, str] # key: path where the possible (never use hyphen(-) or underscore(_) in the name of the file) figure are saved, value: suitable caption for them
    reasonining_info:str # The logic of desgin
    """
    network_details:str # explaining the network structure
    network_paths: List[str]
    plot_paths: Dict[str, str] # key: path where the possible figure are saved, value: suitable caption for them
    reasoning_info:str
    
    def to_dict(self):      
        return self.__dict__.copy() 

@dataclass
class MechanisticModelParams:
    """

    Model parameters for a mechanistic model over network.
    Attributes:
        parameters: Dict[str, float] | dict[str, list[float]] # you can design a mechanistic model with a list of rates or just single rate
        initial_condition_desc: List[str] # randomly selected, or specific initial conditon
        initial_conditions: List[Dict[str, int]]  
        reasoning_info: str # the reasoning and logic for the model and network chosen

    """    
    parameters: Dict[str, List[float]] | Dict[str, float] #
    initial_condition_desc: List[str] 
    initial_conditions: List[Dict[str, int]]  #,
    reasoning_info: str
    
    def to_dict(self):  
        return self.__dict__.copy()

@dataclass
class EpidemicModelInfo:
    """
    Unified result class combining network structure, mechanistic model, and parameters.
    Attributes:
        network_details: str  # From NetworkStructure
        network_paths: List[str]  # From NetworkStructure
        network_reasoning_info: str  # From NetworkStructure
        model_name: str  # From MechanisticModel
        compartments: List[str]  # From MechanisticModel
        transitions: Dict[str, str]  # From MechanisticModel
        epidemic_modeler_reasoning_info: str  # From MechanisticModel
        parameters: Dict[str, List[float]] | Dict[str, float]  # From MechanisticModelParams
        initial_condition_desc: List[str]  # From MechanisticModelParams
        initial_conditions: List[Dict[str, int]]  # From MechanisticModelParams
        params_reasoning_info: str  # From MechanisticModelParams
    """
    network_details: str
    network_paths: List[str]
    plot_paths: Dict[str, str] # key: path where the possible figure are saved, value: suitable caption for them
    network_reasoning_info: str
    model_name: str
    compartments: List[str]
    transitions: Dict[str, str]
    epidemic_modeler_reasoning_info: str
    parameters: Dict[str, List[float]] | Dict[str, float]
    initial_condition_desc: List[str]
    initial_conditions: List[Dict[str, int]]
    params_reasoning_info: str
    def to_dict(self):  
        return self.__dict__.copy()  



@dataclass
class ReflectResult:
    """Class to hold search results."""
    reflection: str
    revise_needed: bool

    def to_dict(self):
        return self.__dict__.copy() 

@dataclass
class SimulationDetails:
    """
    Structure of the output of the code generator agent
    simulation_details:List[str] # describe what have you simulated
    stored_result_path: Dict[str,str] # where do you store the results and their caption
    plot_path:Dict[str,str]  # path to where you store the plots and their caption

    """    
    simulation_details:List[str] # what have you simulated
    stored_result_path: Dict[str,str] # where do you store the results
    plot_path: Dict[str,str] # where do you store the plot
    success_of_simulation: bool # was the simulation successful or not
    reasoning_info:str # reasons and logic for the decisions you made to accomplish the task
    def to_dict(self,iteration=0):
        return self.__dict__.copy()

    def to_json(self,iteration=0):
        pass
        """
        data_dict={}
        data_dict[f'iteration_{iteration}']=self.to_dict(iteration)
        path=os.path.join("output",f'{agent_name}_agent.json')
        with open(path, "w") as f:
            json.dump(data_dict, f, indent=4)
        """
#matplotlib.use('Agg')  # Use a non-interactive backend
@dataclass
class ReflectResult:
    """
    Structure of the output of the code generator agent
    """    
    reflection:str # complete reflection on the generated response   
    revise_needed:bool      # decide if the code needs to be revised,revise_needed= True, or it is completely accurate , revise_needed=False
    def to_dict(self):  
        return self.__dict__.copy()


@dataclass
class SimulationResult(SimulationDetails):
    """
    Structure of the output of the code generator agent
    """
    planning: str #
    def to_dict(self):
        return self.__dict__.copy()


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