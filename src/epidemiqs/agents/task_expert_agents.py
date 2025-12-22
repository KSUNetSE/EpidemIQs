from pydantic_ai import Agent
from typing import List, Dict, Optional, Union
from pathlib import Path
import json
import os
from epidemiqs.agents.tools import CodeExecutor
import requests
from time import sleep
from epidemiqs.utils.llm_models import choose_model
from epidemiqs.utils.utils import log_tokens
from epidemiqs.utils.config import get_settings, Settings
from epidemiqs.utils.long_term_memory import LTM
from pydantic_ai.tools import Tool
from termcolor import colored
from pydantic.dataclasses import dataclass
import asyncio
from pydantic_ai import Agent, BinaryContent
from epidemiqs.agents.prompts import system_prompts
from datetime import date
from termcolor import colored
from tavily import TavilyClient
from pydantic_ai.tools import Tool
from os .path import join as ospj
import traceback

client = TavilyClient(get_settings().apis.TAVILY_API_KEY.get_secret_value() if get_settings().apis.TAVILY_API_KEY else os.getenv("TAVILY_API_KEY"))
@dataclass
class SecretaryAgentResult:
    """Data structure to hold the secretary agent's decision."""
    in_scope_of_EpidemIQs: bool


class SecretaryAgent:
    def __init__(self,
                 llm=None,
                 cfg:Settings=None,
                 name:str="Secretary",
                 system_prompt:str=None,
                 output_type=SecretaryAgentResult,
                 retries:int=20):
        self.system_prompt = system_prompt if system_prompt is not None else system_prompts["secretary_agent"]
        self.name=name
        self.llm=llm if llm is not None else choose_model()["experts"]
        self.secretary_agent=Agent(
            name=self.name,
            model=self.llm,
            system_prompt=self.system_prompt,
            tools=[Tool(self.talk_to_user, takes_ctx=False)],
            output_retries=retries,
            output_type=output_type,
            end_strategy="exhaustive",
            )
        
        self.response=SecretaryAgentResult(in_scope_of_EpidemIQs=False)
        self.memory=None
        
    def talk_to_user(self, message:str):
        """through this function you can communicate with the user directly"""
        print(colored(message,"magenta"))
        print(colored("--------------------------------------------------","magenta"))
        
    async def run(self, query:str):
        full_response = await self.secretary_agent.run(user_prompt=query)
        self._update_memory(full_response)
        self.response = full_response.output
        try:
            while  self.response.in_scope_of_EpidemIQs is False and query.lower() not in ["exit","quit"]:
                try:
                    query=input(colored("Enter your query for secretary agent (type 'exit' or 'quit' to stop): ","cyan"))
                    if query.lower() in ["exit","quit"]:
                        break
                    full_response = await self.secretary_agent.run(user_prompt=query,message_history=self.memory)
                    self.response = full_response.output
                    
                    self._update_memory(full_response)
                    
                except Exception as e:
                    print(colored(f"Error: {str(e)}","red"))
        except Exception as e:
            print(self.response)
            print(colored(f"Error in processing the response: {str(e)}","red"))
        print(colored("\nSecretary Agent has classified your query as in scope of EpidemIQs framework. Proceeding with the EpidemIQs workflow...","green"))
        return self.response
    
    def _update_memory(self, full_response=None):
        try:    
            self.memory=full_response.all_messages()   
        except Exception as e:
            
            print(colored(f"failed to update {self.name}'s memory of : {str(e)}","red"))


class OnlineRetrieverExpert:
    def __init__(self,
                 llm=None,
                 cfg:Settings=None,
                 name:str="OnlineRetrieverExpert",
                 system_prompt:str=None,
                 repo_path:str=ospj(os.getcwd(), "output"),
                 file_name:str="online_retrieved_info.json",
                 ltm_path:str=None):
        self.name=name
        self.cfg=cfg or get_settings()
        self.llm=llm if llm is not None else choose_model(self.cfg)["experts"]
        self.react_module=Agent(self.llm,  #choose_model()["experts"],  
                        name="OnlineRetrieverExpert",   
                        output_retries=self.cfg.workflow.no_retries,
                        system_prompt=system_prompt if system_prompt is not None else system_prompts["online_retriever_expert"],
                        output_type=str,
                        end_strategy="exhaustive",
                        tools=[Tool(self.online_search, takes_ctx=False), Tool(self.get_current_date, takes_ctx=False)]
                        )
        print(colored(f"OnlineRetrieverExpert Agent is ready and powerd by: {self.react_module.model.model_name}", "green"))
        self.error_it=0
        self.file_name=file_name
        self.repo=repo_path
        self.retrieved_information=None
        self.memory=None
        self.ltm_path=ltm_path or ospj(os.getcwd(),"output","LTM.json") 
        self.tokens=0   
    def get_current_date(self, today:bool=True) -> str:
        """Ask and get the current (today) date to use for  up to-date search."""
        print(colored("Getting current date", "green"))
        return f"Today is {date.today()} \n"    
    
    async def online_search(self, query: str, max_results: int = 10)-> str:
        """Search online to for the query to get max of 10 results."""
        print(colored(f"Searching for: {query}", "green"))
        try:
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer="advanced"
            )
            self.error_it=0
            #print(colored(f"Search results:\n{response}", "light_green"))
            self.retrieved_information.append(response)
            return response
        
        except Exception as e:
            self.error_it+=1
            error_trace = traceback.format_exc()
            print(colored(f"Error during search: \n{error_trace}", "red"))
            return f"Error during search:\n {error_trace}\n if error persist more than 3 times (now it is {self.error_it}), it is recommended to use other search query or try again later."

    async def run(self, query: str, max_results: int = 10) -> str:
        """search online to for the query to get max of 10 results."""
        self.retrieved_information=[]
        global total_tokens
        print(colored(f"Sending query to OnlineRetrieverExpert agent:\n{query}", "green"))
        response = await self.react_module.run(f"The user query is:\n {query}.\nToday's date is {date.today()}. Use chain-of-thought to answer accurately and completel using online information\n")
        total_tokens = response.usage().total_tokens
        log_tokens(repo=ospj(os.getcwd(), "output"), agent_name=self.name, llm_model=self.react_module.model.model_name,
                   input_tokens=response.usage().input_tokens, output_tokens=response.usage().output_tokens,total_tokens=total_tokens)
        
        print(colored(f"\nTotal tokens used: {total_tokens}", "green"))
        self.tokens+=total_tokens
        self.memory=response.all_messages()
        self.store_online_info(query,self.retrieved_information, response.output)
        LTM.save_LTM_direct(self.name, self.memory, self.ltm_path)
        return response.output

    def store_online_info(self, query: str, results, agent_response: str) -> None:
        """
        store online retrieval record to a cumulative JSON file.
        
        Args:
            query (str): Scientist query.
            results (Any): The retrieved results .
            agent_response (str): The final summarized or generated response.
        """
        file_path = Path(ospj(self.repo, self.file_name))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
                except json.JSONDecodeError:
                    data = []      
        else:
            data = []

        # append new record
        data.append({
            "query": query,
            "results": results,
            "agent_response": agent_response
        })

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class LiteratureReviewExpert:
    def __init__(self,
                 llm=None,
                 cfg:Settings=None,
                 name:str="LiteratureReviewExpert", 
                 json_file=ospj(os.getcwd(),"output","literature_review.json"), 
                 system_prompt:str=None,
                 sort_type:str="influentialCitationCount",
                 ltm_path:str=None):
        self.name=name
        self.cfg=cfg or get_settings()
        self.llm=llm if llm is not None else choose_model(self.cfg)["experts"]
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = { "X-API-KEY": self.cfg.apis.SEMANTIC_API_KEY.get_secret_value() if self.cfg.apis.SEMANTIC_API_KEY else os.getenv("SEMANTIC_API_KEY") }
        self.json_file = json_file
        self.system_prompt = system_prompt or system_prompts["literature_review_expert"]
        self.results = self._load_json()  
        self.queries=[]
        self.sort_type=sort_type
        self.ltm_path=ltm_path or ospj(os.getcwd(),"output","LTM.json")
        self.react_module = Agent(
            model=self.llm,
            name=self.name,
            system_prompt=self.system_prompt,
            output_retries=self.cfg.workflow.no_retries,
            retries=30,
            end_strategy="exhaustive",
            tools=[Tool(self.conduct_review, takes_ctx=False)],
        )
        self.memory=None
        self.no_paper_per_query=7
        self.max_retrieved_allowed=50
        self.no_retrieved_papers=0
        self.tokens=0
        print(colored(f"LiteratureReview Agent is ready and powered By: {self.react_module.model.model_name}","blue"))

    def _load_json(self):
        """Loads the literature review JSON file or initializes an empty structure."""
        if os.path.exists(self.json_file):
            with open(self.json_file, "r", encoding="utf-8") as file:
                return json.load(file)
        return {"results": []}

    def _save_json(self):
        """Saves the class attribute results to the JSON file."""
        with open(self.json_file, "w", encoding="utf-8") as file:
            json.dump(self.results, file, indent=4)
    def _search_papers(self, query):
        """Search for papers on Semantic Scholar, sorted by importance or relevance.

        Fetches 20 papers from the API, sorts them by the chosen metric, 
        and returns only the top `limit` papers.
        """
        fetch_limit = 100  # max of 100 allowed for a richer selection
        endpoint = f"{self.base_url}/paper/search"
        params = {
            "query": query,
            "limit": fetch_limit,
            "fields": (
                "paperId,title,authors,year,abstract,url,venue,"
                "citationCount,influentialCitationCount,tldr"
            ),
        }

        response = requests.get(endpoint, params=params, headers=self.headers)
        response.raise_for_status()
        data = response.json().get("data", [])
        sleep(1)  # To respect rate limits  
        if not data:
            print(colored(f"No papers found for query: {query}", "red"))
            return []

        # Sort locally by chosen metric (default: influentialCitationCount)
        if self.sort_type in {"influentialCitationCount", "citationCount"}:
            data.sort(key=lambda p: p.get(self.sort_type, 0) or 0, reverse=True)
            print(colored(
                f"Sorted {len(data)} papers for '{query}' "
                f"(sorted by {self.sort_type})", "green"
            ))
        elif self.sort_type == "recency":
            data.sort(key=lambda p: p.get("year", 0) or 0, reverse=True)
            print(colored(
                f"Sorted {len(data)} papers for '{query}' "
                f"(sorted by recency)", "green"
            ))

        else:
            pass
        top_papers = data[:self.no_paper_per_query]
        
        print(colored(
                f"Retrieved {len(top_papers)} of {len(data)} papers for '{query}' "
                f"(sorted by {self.sort_type})", "green"
            ))
        self.no_retrieved_papers+=len(top_papers)


        return top_papers


    def _get_paper_details(self, paper_id):
        """Retrieve detailed information about a specific paper."""
        sleep(1) 
        endpoint = f"{self.base_url}/paper/{paper_id}"
        params = {
            "fields": "title,abstract,authors,year,citations,references,tldr,journal,externalIds,paperId"
        }
        response = requests.get(endpoint, params=params, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        print(colored(f"Papers: {data['title']} By: {', '.join(author['name'] for author in data.get('authors', []))}", "green"))
        return data

    async def conduct_review(self, query: str):
        """Retrieve and store most important papers for a query."""
        if self.no_retrieved_papers >= self.max_retrieved_allowed:
            return "Maximum number of retrieved papers reached."

        if query.lower() in (q.lower() for q in self.queries):
            print(colored(f"Query '{query}' already processed.", "yellow"))
            return next((r for r in self.results["results"] if r["query"] == query), None)

        self.queries.append(query.lower())
        papers = self._search_papers(query)

        if not papers:
            return f"No papers found for query: {query}"

        query_data = {
            "query": query,
            "papers": [],
            "summary": "",
        }

        for p in papers:
            authors = p.get("authors", [])
            formatted_authors = ", ".join(a.get("name", "Unknown") for a in authors[:3])
            if len(authors) > 3:
                formatted_authors += ", et al."
            citation = (
                f"{formatted_authors} ({p.get('year', 'Unknown')}). "
                f"{p.get('title', 'Unknown')}. "
                f"{p.get('venue', 'Unknown') or 'Unknown Journal'}. "
                f"Citations: {p.get('citationCount', 0)}, "
                f"Influential: {p.get('influentialCitationCount', 0)}."
            )

            query_data["papers"].append({
                "title": p.get("title", "Unknown"),
                "abstract": p.get("abstract", "No abstract"),
                "citation": citation,
                "summary (tldr)": p.get("tldr", "No summary"),
                "url": p.get("url"),
            })
            print(colored(f"{formatted_authors}: {p.get('title', 'Unknown')}, {p.get('year', 'Unknown')},{p.get('venue', 'Unknown')} ", "green"))

        self.results["results"].append(query_data)
        self._save_json()
        print(colored(f"Saved {len(query_data['papers'])} papers for '{query}'.", "green"))
        return query_data
    
    async def summarize_results(self,query):
        """ summarize stored literature review results for specific query."""
        if not self.results["results"]:
            print(colored("No literature review data found.", "red"))
            return None

        text_to_summarize = ""

        for i, entry in enumerate(self.results["results"]):
            if entry["query"] == query:
                query_id=i
                text_to_summarize += f"Query: {entry['query']}\n"
                for paper in entry["papers"]:
                    text_to_summarize += f"Title: {paper['paper_title']}\n"
                    text_to_summarize += f"Abstract: {paper['abstract']}\n\n"
                    text_to_summarize += f"Citation: {paper['citation']}\n"


        print(colored(f"Generating summary for Query:\n{query}", "yellow"))
        text_to_summarize+="\nPlease summarize the above papers and provide a comprehensive answer to the query as\n{query}"
        summary = await self.react_module.run(text_to_summarize)
        output_dir = os.path.join(os.getcwd(),"output")
        log_tokens(repo=output_dir, agent_name=self.react_module.name, llm_model=self.react_module.model.model_name,
                   input_tokens=summary.usage().input_tokens,
                   output_tokens=summary.usage().output_tokens,
                   total_tokens=summary.usage().total_tokens)


        self.results["results"][query_id]["summary"] = summary.output  # Add the summary to the results dictionary
        self._save_json()
        self.memory=summary.all_messages()  
        
        print(colored("Summary generated successfully.", "green"))
        return summary.output

    async def talk_to_literature_review_agent(self,prompt:str):
        if self.queries != []:
            for query in self.queries:
                await self.summarize_results(query=query)
        elif self.queries == []:
            print(colored("No queries found, please conduct a review first.", "red"))
            return "No queries found, please conduct a review first."
                
                
        
        response = await self.react_module.run(prompt, message_history=self.memory)
        self.memory=response.all_messages()    
        return response.output  

    async def literature_review(self,query: str, prompt: Optional[str] = None):
        """
        Conduct a literature review by searching for papers and adding them to stored results.
        
        """
        lit_prompt=f"Please search for the following query\n{query} and provide the results in a comprehensive way.\n"
        if prompt is not None:
            lit_prompt+=f"Also consider the Discovery Scinetist prompt\n{prompt}"
        response = await self.react_module.run(user_prompt=lit_prompt, message_history=self.memory)
        self.memory=response.all_messages()
        output_dir = os.path.join(os.getcwd(),"output")
        self.tokens+=response.usage().total_tokens
        log_tokens(repo=output_dir, agent_name=self.name, llm_model=self.react_module.model.model_name,
                   input_tokens=response.usage().input_tokens,
                   output_tokens=response.usage().output_tokens,
                   total_tokens=response.usage().total_tokens)
        LTM.save_LTM_direct(self.name, self.memory, self.ltm_path)
        return response.output
    
class MathExpert():
    def __init__(self,
                 llm=None,
                 cfg:Settings=None,
                 name:str="MathExpert",
                 system_prompt:str=None,
                 repo:str=ospj(os.getcwd(),"output"),   
                 ltm_path:str=None):
        self.name=name
        self.cfg=cfg or get_settings()
        self.llm=llm if llm is not None else choose_model(self.cfg)["mathematician_expert"]
        self.system_prompt=system_prompt or system_prompts["mathematician_expert"]
        self.react_module = Agent(
            model=self.llm,
            name=self.name,
            retries=50,
            system_prompt=self.system_prompt,
            output_type=str,
            tools=[Tool(self.execute_code, takes_ctx=False)],
            end_strategy="exhaustive"
        )
        self.code_executor = CodeExecutor(self.cfg.workflow.time_out_tools)
        self.memory = None
        self.counter= 0
        self.tokens=0
        self.repo= repo
        self.ltm_path=ltm_path or ospj(os.getcwd(),"output","LTM.json")
    async def ask_mathematician_agent(self, question: str) -> str:  
        
        answer_phase1 = await self.react_module.run(question, message_history=self.memory)
        self.memory = answer_phase1.all_messages()
        if self.counter <10:
            prompt = "Now reflect on the answer and use chain of thought to ensure all parts are addressed, and provide concise and to the point answer to how the situation can be validated by simulated by epidemic mechanistic model over static networks."
        else:
            prompt="now reflect on the answer and use chain of thought to make sure it is accurate, precise and addressed all aspectects of the question."
        answer_phase2 = await self.react_module.run(prompt, message_history=self.memory)
        self.memory = answer_phase2.all_messages()
        answer=answer_phase1.output+answer_phase2.output
        self.mathematical_solution=answer
        print(colored(f"Mathematician's answer:\n","magenta"))
        print(colored(answer,"magenta"))
        #await self._save_the_solution()
        self.tokens=answer_phase2.usage().total_tokens
        log_tokens(repo=self.repo, agent_name=self.name, llm_model=self.react_module.model.model_name, input_tokens=answer_phase2.usage().input_tokens, output_tokens=answer_phase2.usage().output_tokens, total_tokens=self.tokens)
        self.counter += 1
        LTM.save_LTM_direct(self.name, self.memory, self.ltm_path)
        return answer

    async def execute_code(self, code: str, return_vars: List[str] = None, script_name:str="reasoning-by-coding.py") -> str:
        """ Execute Python code in a persistent environment.
        Args:
            code: The Python code to execute.
            return_vars: List of variable names to return after execution.
            script_name: Name of the script file to save the code.
        Returns:        
            A string message indicating success or failure and results. 
        """
        return await self.code_executor.execute(code, return_vars, write_to_file=True, script_name=script_name)
    

class DataExpert:
    def __init__(self,llm=None,
                 cfg:Settings=None,
                 name: str="DataExpert",
                 system_prompt:str= None, 
                 output_type:str=str,
                 retries: int = 30, 
                 end_strategy: str = "exhaustive",
                 repo:str=ospj(os.getcwd(), "output"),
                 ltm_path:str=None
                 ):
        
        self.cfg=cfg or get_settings()
        self.llm=llm if llm is not None else choose_model(self.cfg)["experts"]
        self.system_prompt=system_prompt or system_prompts["data_expert"]
        self.name=name
        self.react_module=Agent(
                            self.llm,  # OpenAIModel(_model_name_small, provider="openai") # OpenAIModel(_model_name_main_new, provider="openai") # OpenAIModel(_model_name_small_new, provider="openai") # OpenAIModel(_model_name_small, provider="openai")
                            name=self.name,
                            output_type=output_type,
                            retries=retries,
                            end_strategy=end_strategy,
                            system_prompt=self.system_prompt,
                            tools=[Tool(self.execute_code,takes_ctx=False),] #,Tool(talk_to_extractor_agent,takes_ctx=False)
                                            )                    
        self.memory=None
        self.code_executor=CodeExecutor(self.cfg.workflow.time_out_tools)
        self.repo=repo
        self.counter=0
        self.tokens=0
        self.ltm_path=ltm_path or ospj(os.getcwd(),"output","LTM.json")
    async def forward(self, query: str, data_paths:List[str]) -> str:
        """This function is used to extract data from the CSV file and provide the required information."""
        print(colored(f"\nCalling Data Expert Agent...\n query: {query} \n data_path:\n {data_paths}\n",'blue'))
        
        try:
            self.result=await self.react_module.run(user_prompt=(query+"and data is stored at "+str(data_paths)+" use chain of thought to extract the data you need and always double check to see if the extract results make sense and there is contradiction in the data"), message_history=self.memory)
            self.memory=self.result.all_messages()    
            self.tokens+=self.result.usage().total_tokens
            self.counter+=1
            print(colored(f"\n Data Expert Response {"-"*20}:{self.result.output} \n",'white'))
            log_tokens(repo=self.repo,agent_name=self.react_module.name,llm_model=self.react_module.model.model_name, input_tokens=self.result.usage().input_tokens, output_tokens=self.result.usage().output_tokens, total_tokens=self.tokens)
            LTM.save_LTM_direct(self.name, self.memory, self.ltm_path)
            return self.result.output
        except Exception as e:
            error_trace = traceback.format_exc()
            print(colored(f"Data Expert Agent Could not respond: {str(error_trace)}","red"))  
            return f"Data Expert Agent Is Not Accessibe at the moment,\nError:\n{error_trace}"
    
    async def execute_code(self,code: str, return_vars: List[str] = None, script_name: str ="name-of-script.py") -> str:
        """
        Execute Python code in a persistent environment.
        
        Args:
            code: The Python code to execute.
            return_vars: List of variable names to return after execution.
            write_to_file: if True to write the python code to a file.
            script_name: File name for saving the code.  
            
        Returns:
            A string message indicating success or failure and  the results.
        """
        self.counter+=1
        result =  await self.code_executor.execute(code=code, return_vars=return_vars, script_name=script_name,write_to_file=True)
        if self.counter>25 and self.tokens>200_000:
            result+= "\n\nNote: You are reaching the limit of the number of iterations you can do, please be careful and try to accomplish your task as soon as possible."
        return  result
                
class VisionAgent:
    def __init__(self,
                 llm=None,
                 cfg:Settings=None,
                 name:str="Vision Expert",
                 system_prompt:str=None,
                 retries:int=50              
                 ):

        self.name=name
        self.cfg=cfg or get_settings()
        self.llm=llm or choose_model(self.cfg)["vision_expert"]
        self.system_prompt=system_prompt or system_prompts["vision_expert"] 
        self.react_module= Agent(name=self.name,
                                 model=self.llm,
                                 system_prompt=self.system_prompt,
                                 retries=retries,
                                 end_strategy="exhaustive",
                                 tools=[Tool(self.retrieve_images, takes_ctx=False)]
                                 )
        self.memory=None
        self.tokens=0

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
                    continue  # skip  file and move to the next one 
                
                if not file_name.lower().endswith(supported_formats):
                    prompt.append(f"{file_name} is not in PNG format!!! Only PNG format is supported. "
                                f"Ensure to mention in your response that this file format is not supported!!\n")
                    continue  # skip this file and move to the next one
                
                with open(path, "rb") as image_file:
                    image_data = image_file.read()
                file_name = os.path.basename(path)
                
                prompt.append(f"Image named as:'{file_name}'loaded successfully\n")
                prompt.append(BinaryContent(data=image_data, media_type='image/png'))
                
                
            except Exception as e:
                prompt.append(f"Error reading {path}: {e}")
        #full_prompt="\n".join(prompt)
        return prompt
    async def forward(self, query: str,image_paths: List[str]) -> str:
        """analyze images based on user query and proved inormation"""
        print(colored(f"\nSending prompt to Vision Expert:\nQuery:{query}\nImage Paths:{image_paths}\n", "green"))
        prompt= self.retrieve_images(image_paths)
        prompt[0]=query
        response = await self.react_module.run(user_prompt=prompt, message_history=self.memory)
        self.memory = response.all_messages()           
        #print(colored(f"\nVision Expert Response {"-"*20}:\n{response.output}\n", "gree`n"))
        self.tokens=response.usage().total_tokens
        log_tokens(repo=ospj(os.getcwd(), "output"), agent_name=self.name, llm_model=self.react_module.model.model_name,
                   input_tokens=response.usage().input_tokens, output_tokens=response.usage().output_tokens,total_tokens=self.tokens)
        return response.output
    
    
if __name__ == "__main__":
    import asyncio, os, json
    from datetime import datetime
    from termcolor import colored

    log_path = os.path.join(os.getcwd(), "output", "agent_test_log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    async def test_math_expert():
        print(colored("\n testing MathExpert  ", "cyan"))
        mexp = MathExpert()
        print(colored(f"{mexp.name} Agent is ready and powered by: {mexp.react_module.model.model_name}", "green"))
        res = await mexp.ask_mathematician_agent(r"`"" Using CoT to answer: Given an uncorrelated random network (configuration model) with mean degree z = 3 and mean excess degree q = 4, and an epidemic with R0 = 4, what is: The critical vaccination fraction if vaccination only targets nodes with degree exactly k=10? Please provide """)
        print(res)
        
    async def test():
        tests = []
        try:
            print(colored("\n--- Testing SecretaryAgent ---", "cyan"))
            sec = SecretaryAgent()
            res = await sec.run("Is epidemic modeling part of EpidemIQs?")
            tests.append({"agent": "SecretaryAgent", "result": res.__dict__})
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(colored(f"Error during SecretaryAgent test:\n{traceback_str}", "red"))
            tests.append({"agent": "SecretaryAgent", "error": str(e)})

        try:
            print(colored("\n--- Testing OnlineRetrieverExpert ---", "cyan"))
            ore = OnlineRetrieverExpert()
            res = await ore.run("recent LLMs for epidemic modeling")
            tests.append({"agent": "OnlineRetrieverExpert", "result": res[:300]})
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(colored(f"Error during OnlineRetrieverExpert test:\n{traceback_str}", "red"))
            tests.append({"agent": "OnlineRetrieverExpert", "error": str(e)})

        try:
            print(colored("\n--- Testing LiteratureReviewExpert ---", "cyan"))
            lit = LiteratureReviewExpert()
            res = await lit.literature_review(query="epidemic forecasting with LLMs", prompt="Provide a comprehensive literature review on the topic.")
            tests.append({"agent": "LiteratureReviewExpert", "papers": len(res.get('papers', [])) if isinstance(res, dict) else str(res)})
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(colored(f"Error during LiteratureReviewExpert test:\n{traceback_str}", "red"))
            tests.append({"agent": "LiteratureReviewExpert", "error": str(e)})

        try:
            print(colored("\n--- Testing MathExpert ---", "cyan"))
            mexp = MathExpert()
            res = await mexp.ask_mathematician_agent("derive basic reproduction number R0 in SIR model")
            tests.append({"agent": "MathExpert", "result": res[:300]})
        except Exception as e:
            traceback_str = traceback.format_exc()  
            print(colored(f"Error during MathExpert test:\n{traceback_str}", "red"))
            tests.append({"agent": "MathExpert", "error": str(e)})

        try:
            print(colored("\n--- Testing DataExpert ---", "cyan"))
            dexp = DataExpert()
            res = await dexp.forward("compute mean infection rate", ["output/results-00.csv"])
            tests.append({"agent": "DataExpert", "result": str(res.output)[:200] if hasattr(res, 'output') else str(res)[:200]})
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(colored(f"Error during DataExpert test:\n{traceback_str}", "red"))    
            tests.append({"agent": "DataExpert", "error": str(e)})

        try:
            print(colored("\n--- Testing VisionAgent ---", "cyan"))
            vexp = VisionAgent()
            res = await vexp.forward("analyze infection heatmap", ["sample.png"])
            tests.append({"agent": "VisionAgent", "result": res[:200]})
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(colored(f"Error during VisionAgent test:\n{traceback_str}", "red"))   
            tests.append({"agent": "VisionAgent", "error": str(e)})

        # log results
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- Test run {datetime.now()} ---\n")
            json.dump(tests, f, indent=2)
        print(colored(f"\nAll results saved to {log_path}", "green"))

    asyncio.run(test_math_expert())