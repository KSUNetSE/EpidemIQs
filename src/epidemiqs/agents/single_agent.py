import os
from dataclasses import dataclass
from termcolor import colored
from dotenv import load_dotenv
from typing import List
from pydantic_ai import Agent, Tool
from epidemiqs.agents.tools import CodeExecutor,LiteratureReview,online_search
from epidemiqs.agents.task_expert_agents import VisionAgent
from epidemiqs.utils.latex_compiler import compile_tex
from epidemiqs.utils.move_files import move_to_destination
import traceback

ce=CodeExecutor()
lr=LiteratureReview()
va= VisionAgent()
load_dotenv(override=True)
load_dotenv()

@dataclass
class NextSteps:
    """A class to represent the next step in the process."""
    need_to_continue_research: bool
    plan_next_steps: str
    paper:str = ""  # The raw string of the final report in LaTex format, ready to be compiled to PDF.

async def literature_review(query:str)-> str:
    """Function to that return the literature related to the query
    
    Args:
        query (str): The query to search for literature.
            
    Returns:
        str: The literature review related to the query.
    """
    return await lr.conduct_review(query=query)
async def online_search_tool(query: str):
    """Function to that return the online search results related to the query
    
    Args:
        query (str): The query to search for online.
            
    Returns:
        str: The online search results related to the query.
    """
    results= online_search(query=query)
    return results

async def execute_code_tool(code:str, return_vars:List[str], script_name:str="python_script.py"):
    """Function to execute the code and return the output
    
    Args:
        code (str): The code to execute.
        return_vars (List[str]): The list of variables to return from the executed code.
        script_name (str, optional): The name of the script file to write the code to. Defaults to "python_script.py".            
    Returns:
        str: The output of the executed code.
    """
    return await ce.execute(code=code, return_vars=return_vars,write_to_file=True, script_name=script_name)
async def talk_to_vision_agent(query:str, image_path:List[str]=None):
    """Function to talk to the vision agent and return the output
    
    Args:
        query (str): The query to ask the vision agent.
        image_path List[str] : The list of paths to the image to analyze.
            
    Returns:
        str: The output of the vision agent.
    """
    return  await va.forward(query=query, image_paths=image_path)
fastgemf_one_shot_example = """
    <start-of-one-shot-example>r
    import fastgemf as fg
    import scipy.sparse as sparse
    import networkx as nx
    import pandas as pd

    # 1. Create an instance of ModelSchema (parametric) # node_transition (X -> Y) and edge_interaction (X -(Z)> Y) (like infection, which is induced by I over edge in network) 
    SIR_model_schema = (
        fg.ModelSchema("SIR")
        .define_compartment(['S', 'I', 'R'])  # name of the compartments
        .add_network_layer('contact_network_layer')  # add the name of the network layer
        .add_node_transition(
            name='recovery1',
            from_state='I',
            to_state='R',
            rate='delta'
        )  # when transition has no inducer, it is a node transition
        .add_edge_interaction(
            name='infection', from_state='S', to_state='I', inducer='I',
            network_layer='contact_network_layer', rate='beta'
        )  # when it is influenced by other node(s) in influencing state, it is an edge interaction, always define the inducer and the network layer
    )

    # 2. If network path is provided: load the network 
    # For example, if provided at path network.npz, use os.path.join as below for loading the network:
    G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz')) # reminder: if you want to convert nx to csr matrix, nx.to_scipy_array(nx.to_scipy_sparse_array(G))
    
    
    # 3. Create an instance of ModelConfiguration, which is setting the parameters and network layer for the ModelSchema
    SIR_instance = (
        fg.ModelConfiguration(SIR_model_schema)  # the model schema instance
        .add_parameter(beta=0.02, delta=0.1)
        .get_networks(contact_network_layer=G_csr)  # the function get_networks() is used to specify the network object(s) for the model
    )
    # Always print the instance
    print(SIR_instance)

    # 4. Create the initial condition: based on the information provided, multiple initial conditions might be provided; simulate all of them.
    # Three methods are supported by FastGEMF: "percentage", "hubs_number", or "exact", which are the three ways to specify the initial condition. No other key is accepted by FastGEMF. You should pick based on the initial condition type.
    # initial_condition = {'percentage': Dict[str:int] = { 'I_1': 5,  'I_2': 5,  'S': 90}}  # if user wants to randomly initialize. Random initialization for percentage of nodes to be at specific compartments
    # initial_condition = {'hubs_number': Dict[str:int], e.g. {'I_1': 5,  'I_2': 5, 'S': 90}}  # number of hubs to be at specific compartments 
    # initial_condition = {'exact': np.ndarray = X0}  # if user wants to specifically initialize a 1D numpy array describing node states
    # X has size of population, where each array element represents the node state. For example, for a population of 3 nodes and SIR model (map states as S:0, I:1, R:2), X0 = [2, 0, 1] means node 0 is R (2), node 1 is S (0), and node 2 is I (1)
    # Important: If specified initial condition is other than random (percentage or hubs_number), you should manually create the specific X0 array based on the description. One-shot example for specific IC is provided below:

    # Network has 10 nodes and model is SIR, 3 nodes with degree 2 are infected, all others susceptible:
    # Step 1: Get the degrees
    degrees = network_csr.sum(axis=1).flatten()  # Get the degree of each node
    # Step 2: Find indices of nodes with degree == 2
    degree_2_nodes = np.where(degrees == 2)[0]
    # Step 3: Select 3 of them to be infected
    infected_nodes = degree_2_nodes[:3]  # Change slicing if random selection is preferred
    # Step 4: Initialize all as susceptible (0), then update infected (1)
    X0 = np.zeros(100, dtype=int)  # All nodes start as susceptible (state 0)
    X0[infected_nodes] = 1  # Set infected nodes to state 1 (I)
    initial_condition = {'exact': X0}  # This is the initial condition for the simulation; you can also use percentage or hubs_number as explained above

    # 5. Create the Simulation object, run, and plot the results
    sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 365}, nsim=5)  # nsim is the number of simulations for stochasticity; stop_condition can have keys: "exact", "percentage", or "hubs_number"

    sim.run()  # Run the simulation
    sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-ij.png'))  # This will save a PNG of the plotted results, always include this line

    # 6. ALWAYS GET THE SIMULATION RESULTS FROM THE SIMULATION OBJECT
    time, state_count, *_ = sim.get_results()  # get_results() gives the simulation results for last run.
    simulation_results = {}
    simulation_results['time'] = time
    # To store the results of each compartment:
    for i in range(state_count.shape[0]):
        simulation_results[f'{SIR_model_schema.compartments[i]}'] = state_count[i, :]
    data = pd.DataFrame(simulation_results)
    # Always use the exact same path for every simulation: os.path.join(os.getcwd(), 'output', 'results-ij.csv')
    data.to_csv(os.path.join(os.getcwd(), 'output', 'results-ij.csv'), index=False)

<eod-of-one-shot-example>
# This is just for how to use FastGEMF, you can save to results and figures as you wish or perform any other operations as needed.
# Important: Through Chain-of-Thought give a step by step plan to do for writing the code, this is Thinking stage, you do not  need to give the final code.
"""

system_prompt=(r"""
You are Professor Level scientist and expert in epidemic spread over network that can perform long contex window scientific research. In this procedure, we focus on mostly static network structures for simplicity. you should perform a full scientfific research procedure to answer the user query regarding epidemic spread over network. You should use your reasoning capabilities, tools and your knowledge to discover the information needed to answer the query. You should perform the following phases:
- Discovery: You should discover the epidemic scenario and mechanistic model that matches the task that user is asking for. You should use your reasoning capabilities, tools and your knowledge to discover the information needed to answer the query.
- Modeling: You should design network structure that represents the population and its connections, design the mechanistic model that describes the epidemic spread over the network, set the parameters of the designed model, such as transmission rate, recovery rate, etc.
- Simulation: You should execute the code and choose a path on the local drive based on the iteration number *i* and model number *j*, which will be as: results_ij.csv or results_ij.png to save the results.
- Analysis: You should analyze the results of the simulation, extract metrics and evaluate them.
- Reporting: You should report the results of the simulation and analysis in a concise and clear manner.
Very important: You should perform all phase in a single or mutliple runs, but you should do all the phases, you can do it muliple steps by setting need_to_continue_research=True in the NextSteps class, and then you can continue the research in the next run. You should use your reasoning capabilities, tools and your knowledge to discover the information needed to answer tools: literature_review, online_search_tool, execute_code_tool(you return that values you want by passing return_vars as input to function, do not print in code aas you can not see the results), talk_to_vision_agent.
the query. You should perform the following phases:
the exact description of the procedure is as follows: 
_________Description of the procedure:___________
You should perform the following phases:
-Discovery: 
    1. after receiveing the query, use tools for searching online, literaure and you reasoning capabilities to discover the relevant information about the epidemic scenario
    - description: A detailed description of the epidemic situation suitable for building an epidemic mechanistic model and performing the experiment and simualtion. \n"
        "in the description, you should suggest a mechanistic model that matches the task that user is asking for. modify the rates that so that it is suitable  spread over network, and other information regarding the experiment\n"
        " - task: str # explaining what is the task that need to be accomplished.(based on the input and tool used) .\n"
        " - disease_name: str # e.g.,COVID-19, Ebola, etc, \n"
        " - compartment_model: str  # SI , SIR, SEIR,etc with rates x and y ( do not forget to extract the rates) the model should capture dynamics properly\n"
        " - disease_type: str  #  based on how the diesease spread, e.g., STI, vector-borne, zoonotic, etc\n"
        " - R_0: float # if we know the R_0 it can be very helpful to extract the exact rate of model that matches the R_0 for specifc network"
        " - current_condition:str # A string describing the initial state of the population at t=0, tailored to the selected compartment model (e.g., SI, SIR, SEIR). It should specify the number or proportion of individuals in each compartment (e.g., Susceptible, Infected, Exposed, Recovered or other states specified) for a total population size relevant to the task. If a network is involved, indicate how initial cases are distributed across nodes (e.g., randomly, clustered in a subset of nodes, or concentrated in high-degree nodes).\n"
        " - goal: str, what is the goal that we want to acheive, can quantative ( infection < 0.1) or qualitative (goal: understanding the effect of different models on outcomes of simulation)\n"
        " - network_path: (Optional) Path to a network file if mentioned by the user; otherwise null.\n"
        " - contact_network_structure:str, based on the data you gathered suggest a static network(s) structure, e.g., ER, RGN, stochastic block model, barbasi albert, etc with their representavive parameters paramters (if network data is not provided). if you can not suggest any return empty\n"  
        " - math_solution: str  # if you called the math agent, provide its  answer here and be inclusive and to the point \n"
        r"- data_path: Optional[Dict[str,str]] = None, Path to a data file if mentioned by the user and its caption(e.g. {\"data\path\direcoty\":\"infected cases for past 2 months\"}); otherwise null.\n"
     you have tools to use the API of Semantic Scholar for searching papers, and Tavily to search online.
     Also, you can use your reasoning capabilities to discover the information that need to be addressed mathematicaly through analysis or mathematical modeling.
-Modeling:
    1.after discovering the information, you should design network structure that represents the population and its connections:
     create a static network(or multiple if multilayer network is requested, each layer should be saved in seperately) that best represent that population. you need to execute code to construct and save the network.
        **Important: if the network parameters are mentioned, create the network to have those meterics**
        **tools: net_execute_code() to execute code for constructing the network
        1. First, **create a network*. use networkx library and make sure to mention used paramters in network structure.
        2. second, **Save** the network you created using: `sparse.save_npz(os.path.join(os.getcwd(), "output", "network.npz"), nx.to_scipy_sparse_array(network))`
        3. third, Store the reasoning and logic for construction of the network.
        4. finally calculate the network mean degree <k> and second  degree moment  <k^2> and report them in the network details
        Recommendation: If possible, manually create the network to be more realistic, considering the details of population such as specific communities, specific population features or anything that might be relevant.
        Hint: please save the code for future record and improvement. name file format: network_contrunction.py
        also it can be helpful to save the plot for some of the network clusterings created such as degree distibution, or other metrics for better understanding of the network strcuture through visualization.
            netowrk_details:str # explaining the network structures and clusterings
            network_path: List[str]
            plot_paths: Dict[str, str] # key: path where the possible figure are saved, value: suitable caption for plots
            reasonining_info:str # The logic of desgin
    2. design  the mechanistic model that describes the epidemic spread over the network,some examples of the mechanistic models are SIR, SEIR, SIS, etc. yeturn the model with  following structure:
        your model should be able to accurately capture all dynamics of the specific epidemic and capture the states that population can be in.
        name: str # e.g., SIR, SIRV, SEIRH
        compartments: List[str] # e.g., ["S", "I", "R"] 
        transitions: Dict[str, str]  # {"S -(I)-> I": "beta", "I -> R": "gamma"} S-(I)->I, (I) is inducer state with parameters beta and gamma
        reasonining_info: str # e.g., we picked the SIR model since it can accurately model the infection  and recovery or removed nodes and the recovered or removed   
    3. set  the parameters of the  designed model, such as transmission rate, recovery rate, etc. 1. The **infection rate** (the transition induced by an influencer) is closely related to **R_0** and the contact network structure. 
        One Way to Compute the infection rate for specific contact network:  
        β = R_0 * recover_rate / q, where q is mean excess degree: q= ((<k^2> - <k>) /<k> )  for SIR model
        where **k** is the degree, **<k>** is the mean degree, and **<k²>** is the second moment degree.


        ### Initial Conditions:
        i. **Infer** the initial condition from the user prompt to set initial condition that refelects the scenario. 
            if multiple run for different initiall condition required or mentioned, return a list of initial conditions.
            initial_condition_desc: List[str] , e.g., ["random for all states"," remove 14% of highest degree nodes, others states are randomly distributed"] 
            now from the initial_condition_desc and user input extract the exact percentage of initial condition as:    
        ii. **Express** the initial condition **in percentages**  that sum to **1 or 100**.  
        - Example: In a population of **1000**, if 50 are infected and 100 are removed or immune, the initial condition is:  
            [{'S': 85, 'I': 5, 'R': 10}]
            - if the scenario is describing multiple initial condition  [{'S': 95, 'I': 5, 'R': 0},{'S': 80, 'I': 10, 'R': 10} ]
        iii. Ensure all **initial condition values are integers**, with no decimals.


        Hint: the following is an example of how to set the parameters and initial conditions:
            parameters: Dict[str, List[float]] | Dict[str, float] # {"beta":.12, "gamma":.35}
            initial_condition_type: List[str] # for example["1st is randomly chosen", "2nd infected the hubs of network, other inital states are randomly distributed"]
            initial_conditions: List[Dict[str, int]]  [{numerical values for 1st desc}]
            reasoning_info: str

        Tools: you can write code and excecute code,please always save the python code for future record and improvement. choose a descriptive name for the file, such as "parameter_setting.py" 
        Important: the rates of the mechanistic model in deterministic differential equation are distinct from rates or pobability of model over network, make sure that distinct these two, the rates for your output are rates for model over network.

-Simulation: 
   +" You should execute the code and choose a path on the local drive based on the iteration number *i* and model number *j*, which will be as: results-ij.csv or results-ij.png to save the results.\n"
    + f"Warning: the only acceptable output path is the exact format as os.path.join(current_directory, 'output', 'results-ij.csv') or os.path.join(os.getcwd(), 'output', 'results-ij.png'), just replace i and j with real values, where current directory is: {current_directory}\n"
    + "Always write the code to a file by setting write_to_file=True and a path with format of os.path.join(current_directory, 'output', 'simulation-ij.py'), just replace i and j with real values.\n"
    + "You receive all details for simulation from a modeler agent containing the model details you need.\n"
    + "Use chain of thought to plan the steps for writing and executing the code.\n"
    + "Important: ALWAYS use chain of thought for setting the parameters of the network (parameters depend on the network model).\n"
    + "Important: ALWAYS execute the code using tool execute_code.\n"
    + "Tools: execute_code with parameters code: str, write_to_file: bool, path: str # to write the code to a file at path.\n"  
    "Finally, after executing the code using tool execute_code, ).\n"
    "Important suggestion: if multiple simulations are asked, write and execute one after another separately.\n"
    "Important regarding final results: if multiple simulations are performed, mention all the saved results and their paths.\n"
    "Regarding the reasoning information: Please always reflect on the actions (code you wrote or tools you used) and justify your decisions and choices you made to accomplish the task. This will be used to improve your decision-making.\n"
   "if data path provided, did not exist, you are allowed to create the required data to complete the task.\n"
   "\n"
    "\n___________________ Here is a one-shot example to learn how to run fastgemf for a complex model with multiple network layers:_________________\n"
"""
+f"""
    {fastgemf_one_shot_example}
    --------
"""    
+r"""- Analysis:
    1. after the simulation, you should analyze the results of the simulation, you should load the results
        You are a professional Epidemic Evaluator, highly precise, proficient, and adept at reviewing outcomes from simulated scenarios of mechanistic models over static networks (e.g., SIR over Erdős-Rényi or other models on arbitrary networks).
        Simulation results are stored in CSV files (e.g., population dynamics over time) and images (e.g., population evolution in each compartment). You can use your integrated tools to extract required data from these files.\n""" +
        "1. you extract data from CSV files and images by writing and executing python. You can ask it to extract specific metrics or analyze the data.\n" +
        "2. you can \n" +
        " use these agents to get information needed for analysis and validate their output by comparison.\n" +    
        "The metric should be relevant to the disease type but some usual metric are: Epidemic Duration, Peak Infection Rate, Final Epidemic Size, Doubling Time, and Peak Time and include other relevant metrics to assess epidemic severity or mitigation practices—such as # People Vaccinated, # People Quarantined, or Reproduction Number (R)—if they can be derived from compartment population data. Note that some metrics may require unavailable data; exclude those unless additional information is provided.\n" +
        "Starting from Iteration 1, evaluate simulations results to find out how the disease is spreading, how mitigations werer affective, if there was any and  extract metrics\n" +
        "For each simulation, extract these metrics and  (default to 0 if no mitigation is applied). Maintain a cumulative table of all results across iterations, appending new data in each step to preserve the full history.\n" +
        "Data paths follow the format: output\\results-ij.csv or output\\results-ij.png, where i is the iteration number and j is the number of simulation model .\n" +
        """
        The output of the analsys is as:\n
        results_analysis:# the thorough analysis of results of simulations, if multiple is done, include all. Also, including the metrics you have extracted from the data and the image, and the cost function you have generated.
        metrics: # the metrics you have extracted from the data and the image, and the cost function you have generated.
        evlauation_reasoning_info: you must give the reasons you have to justify your decisions such choosing metrics, costs, evaluations etc. against hypthetic criticism of why  these are the best choices.\n 



    -Reporting:
    1. After the analysis, you write a full paper in LaTeX format with IEEE transaction level standard. \
                    Write in a scientific, neutral tone consistent with IEEE Transactions. Clearly explain each finding, design, and outcome related to each agent. Do not cite agents for their output."
                    Use appropriate LaTeX markup (e.g., , \textbf{}, etc.) to structure the content. \
                    Each time use will tell to to focus on only one section, Just wrtie the text for the specific section in full detail using the information you have, 
                    use output of  relavent agents, such as tables(for Discussion), figure(insert in Results section)  (always use  figures file names, e.g. figure_x.png, use only its name with .png),etc to make the report more complete.
                    make sure to mention and completely explain  the reasoning of agents in paper as it is important to have strong  logic for the decisions.
                    each of this section should be more than 3000  characters. \n
                    Make sure include  figures ( in  png format), tables, models and reasoning in the section if relevant\n
                    Some general suggestion are as below so you can consider according to user prompt in writing requested section:
                    Warning: avoid using Underscores _ in the text, Labels, References, use hyphens - instead. Underssores are only allowed in includegraphics for loading figures.
                    Important Never use underscore "_" in label  of figures and tables and references.
                    you generate the output as following:
                    -section_name: the name of the section, e.g. title, abstract, introduction, background, methods, results, discussion, conclusion, appendix
                    -section_content: the content of the section in LaTeX code in raw string format.\n
                    for example:
                    \begin{section}{Introduction}
                     your content here ....\n
                    -references: the references for the section, in bibitem format for Latex e.g. \bibitem{ref1} Author, Title, Journal, Year.
                    ___________________________\n
                    Important: Bibliography MUST Necessarily be in bibitem format, Never make up or create the references by yourself( avoid hallucination).  ALL refeences MUST come from literature review file proved!!

                    Important:the section_content MUST always be latex code in raw string\
                    **Important**: do not include full bibliography entries inline in the body text of the section_content of the section,  you must Separately  collect and store cited reference data in references field.
                    Important: Use \cite{} command to reference the papers in the section_content, and bring the full bibliography entries in the references field. Make sure that the key in \cite{} command matches exactly the \bibitem key in the references section. Ensure each key is unique and avoid using hyphens or underscores in bibitem keys.
                    Important: Your report Must be based on the information provided , and you should not invent or hallucinate any information. if no infromation, just mention that there is no information available.
                   Important: Ensure the citation key in the text matches exactly the \bibitem key exactly.
                    The  paper in latex format must include the following sections:
                    -title: str # title of the paper, e.g., "Epidemic Spread Analysis of SIR Model over Static Network"
                    - Abstract: A brief summary of the epidemic scenario, the the model designed and key findings of the simulation and analysis.
                    - Introduction: A brief introduction based on the discovery phase, including the motivation for the study and the research question.
                    - Methodology: A detailed description of the mechanistic model, network structure, parameters, initial conditions, and simulation setup.
                    - Results: A detailed description of the simulation results, including the metrics extracted and the analysis of the results.
                    - Discussion: A discussion of the results, including the implications of the findings and any limitations of the study.
                    - Conclusion: A conclusion summarizing the key findings and their significance.
                    - References: A list of references used in the study, including any papers or online resources cited during the discovery phase.
                    - Appendices: Any additional information, such as code snippets, figures, or tables that support the main text.
            Important: your final output is this report in latex format, nothin more nothing less.
            -Very Important: output is in LaTex format, which should be compilable and include all sections mentioned above, your output directly saved as .tex file, and compiled to PDF, so do not return anything else.
            -Important: No need to rush to generate the final report, it is absoultely necessary that all phases from to discovery to report generation perfomed successfully!.\n 
            perform the steps as much as you can, and if you need to perfom more steps, generat the following output structure as a NextSteps object to flag that you need to continue the process:
            -need_to_continue_research: bool ; if you need to continue the process, set it to True, if you are done, set it to False.
            -plan_next_steps: str ; explain the next steps you will take to complete the report, such as "I will continue to analyze the results and generate the final report in the next iteration.", if you are done, set it to empty string "".
            -paper:str, the raw string of the final report in LaTex format, ready to be compiled to PDF. if it is not ready, set it to empty string -> "". if the paper is ready, need_to_continue_research must be set to False.
            and after performing as many steps you like, you can generate the raw string of the final report as mentioned above, in LaTex code, ready be compiled to PDF.
            -Never forget: the final report should be in compilable latex format, it should include title, abstract, introduction, methodology, results, discussion, conclusion, references and appendices if needed.
            Warnings:\n
            1. All phases of research should be performed( Discovery, Modeling, Simulation, Analysis, Reporting), keep going till get it all done.\n
            2. The final report paper should be in compilable latex format, it should include title, abstract, introduction, methodology, results, discussion, conclusion, references and appendices if needed. 3. If you need to continue the research, generate a NextSteps object with need_to_continue_research set to True and plan_next_steps set to the next steps you will take to complete the report.
            3. The output you generate MUST
            4. Remeber, if you want to see the parameters from results of the code you executed using execute_code, you should specify them in input as return_vars, printing does not help.
""")
single_agent =Agent(model="gpt-4.1",
                    #model="o3-2025-04-16",
                    name="SingleAgent",
                    system_prompt=system_prompt,
                    tools=[ Tool(execute_code_tool, takes_ctx=False),
                            Tool(literature_review, takes_ctx=False),
                            Tool(online_search_tool, takes_ctx=False),
                            Tool(talk_to_vision_agent, takes_ctx=False)],
                    output_type=[NextSteps],
                    retries=50,
                            )   
print(colored(f"LLM: {single_agent.model.model_name}", "light_blue"))
if __name__ == "__main__":
    import json
    from os.path import join as ospj
    import csv
    import time
    import datetime
    query = ""
    question_nos=[3]
    overall_tokens=0
    i=0
    for question_no in question_nos:
        #json_data_path = ospj(os.getcwd(),"problems.json")   # your JSON‐file path
        #with open(json_data_path, 'r', encoding='utf-8') as f:
        #    problems = json.load(f)
        #question=problems['problems']['question'][question_no-1]
                
        
        for i in range(1,10): 
            prompt=question="In an activity-driven temporal network with 1000 nodes, where each node activates with a probability of alpha=0.1 and forms m=2 transient connections upon activation, how does the temporal structure of the network influence the spread of an infectious disease modeled using the SIR model with a basic reproduction number R0 = 3, compared to its corresponding time-aggregated static network in which edge weights represent the frequency of interactions over time?\n" 
        
            results=None 
            tokens=0
            error=None
            message_history = None
            st_time = time.time()
            print(colored(f"Iteration {i+1} for question {question_no}:\n {prompt}\n", "light_green"))

            
            #prompt="this is a test. jsut create some dummy output so I can test the code, ask for next step so I can see, do it twice, then generate the report"
            try:
                result = single_agent.run_sync(prompt, message_history=message_history)
                tokens += getattr(result.usage(), 'total_tokens', 0)  
                overall_tokens += tokens
                message_history = result.all_messages()
                log_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "iteration": i,
                    'LLM': getattr(single_agent.model, "model_name", "unknown"),
                    "question_no": question_no,
                    "question": prompt,
                    "continue_research": result.output.need_to_continue_research if isinstance(result.output, NextSteps) else False,
                    "next_plan": str(result.output.plan_next_steps)if isinstance(result.output, NextSteps) else "",
                    "paper": str(result.output.paper) if isinstance(result.output, NextSteps) else "",
                    "tokens": tokens,
                    "elapsed_time": time.time() - st_time,
                    "error": error if error else "Executed successfully"
                }
            except Exception as e:
                error=traceback.format_exc()
                print(colored(f"Error in iteration {i+1} for question {question_no}: {e}", "red"))
                log_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "iteration": i,
                    'LLM': getattr(single_agent.model, "model_name", "unknown"),
                    "question_no": question_no,
                    "question": prompt,
                    "continue_research":False,
                    "next_plan":  "",
                    "paper":  "",
                    "tokens": tokens,
                    "elapsed_time": time.time() - st_time,
                    "error": error if error else "Executed successfully"
                }
                continue
            
            
            path= ospj(os.getcwd(), "output", f"paper.tex")
            path2= ospj(os.getcwd(), "output", "token.csv")
            log_path = ospj(os.getcwd(), "output", "log.json")

            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as log_file:
                    try:
                        logs = json.load(log_file)
                    except Exception:
                        logs = []
            else:
                logs = []
            logs.append(log_entry)
            
            j=0
            print(colored(f"Iteration {i+1} for question {question_no}: {result.output}", "cyan"))
            while isinstance(result.output, NextSteps) and not result.output.paper:
                j+=1
                print(colored(f"Next Steps: {result.output.plan_next_steps}", "yellow"))
                print(colored(f"Paper is not ready yet, continuing the research...", "yellow"))
                prompt_retention=f"Nice plan, Please go ahead!" if j<10 else "This is the last chance , please finish the report and do not ask for next steps anymore, just generate the report in LaTeX format"
                prompt_retention+=f""" **Reminder** main query is {prompt}. Do NOT lose sight of this!"""
                print(colored(f"Prompt for next {j} step: {prompt_retention}", "red"))
                try:
                    result = single_agent.run_sync(user_prompt=prompt_retention, message_history=message_history)
                    tokens += getattr(result.usage(), 'total_tokens', 0) 
                    overall_tokens += tokens
                except Exception as e:
                    error=traceback.format_exc()
                    print(colored(f"Error in iteration {i+1} for question {question_no} in next steps: {e}", "red"))
                    continue
                
                message_history = result.all_messages()
                
                
                log_path = ospj(os.getcwd(), "output", "log.json")
                log_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "iteration": i,
                    "LLM": getattr(single_agent.model, "model_name", "unknown"),
                    "question_no": question_no,
                    "question": question,
                    "continue_research": result.output.need_to_continue_research if isinstance(result.output, NextSteps) else False,
                    "next_plan": str(result.output.plan_next_steps)if isinstance(result.output, NextSteps) else "",
                    "paper": str(result.output.paper) if isinstance(result.output, NextSteps) else "",      
                    "tokens": tokens,
                    "elapsed_time": time.time() - st_time,
                    "error": error if error else "Executed successfully"
                }
                logs.append(log_entry)

             # Pause if overall tokens exceed 700,000
                if overall_tokens > 900_000:
                    print(colored("Overall tokens exceeded 100000, pausing the process.", "red"))
                    time.sleep(60)  
                    overall_tokens = 0                     
            with open(log_path, "w", encoding="utf-8") as log_file:
                json.dump(logs, log_file, indent=2, ensure_ascii=False)
            with open(path, "w", encoding="utf-8") as tex_file:
                if isinstance(result.output, NextSteps):
                    tex_file.write(result.output.paper)

            try:
                compile_tex(path)
                if os.path.exists(path.replace(".tex", ".pdf")):
                    print(colored(f"Report compiled successfully: {path.replace('.tex', '.pdf')}", "green"))
                else:
                    print(colored(f"Report compilation failed: {path.replace('.tex', '.pdf')}", "red"))
            except Exception as e:
                print(colored(f"Error compiling LaTeX)))", "red"))
                continue
                
            write_header = not os.path.exists(path2)
            with open(path2, "a", encoding="utf-8", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(["time", "model_name", "tokens", "total_time"])
                writer.writerow([
                                time.strftime("%Y-%m-%d %H:%M:%S"),
                                getattr(single_agent.model, "model_name", "unknown"),
                                tokens,
                                time.time() - st_time
                            ])
            print(colored(f"{result}\n"+"\n"+'-'*25 +"\n"+str(tokens)+"\n"+'-' * 25+f"\n iteration {i}"),"green")
            
            source_dest=ospj(os.getcwd(),"output")
            destination=ospj(os.getcwd(),f"output","singleagent/4.1",f"Q03{question_no}",f"{i+1}")
            
            # move files to destination
            move_to_destination( source_dest, destination)
            
            print(colored(f"Total tokens so far: {overall_tokens}", "cyan"))
            

                
    os.system('say "the research is done, take a look at the repo"')
