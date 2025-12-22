from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from typing import List
from datetime import date
from termcolor import colored
import openai
from tavily import TavilyClient
from pydantic_ai.tools import Tool

import os
from os .path import join as ospj
from epidemiqs.utils.utils import log_tokens
from epidemiqs.utils.config import get_settings
import traceback
from dataclasses import dataclass
from epidemiqs.utils.llm_models import choose_model

cfg=get_settings() 
client = TavilyClient(get_settings().apis.TAVILY_API_KEY.get_secret_value() if get_settings().apis.TAVILY_API_KEY else os.getenv("TAVILY_API_KEY"))
_llm_model=choose_model()["experts"] 

@dataclass
class SearchResults:
    """ hold search results."""
    query: str
    results: str
    def to_dict(self):
        return self.__dict__.copy()
online_retriever_expert_system_prompt =(
f"""You are a sharp data extrater agent from the web that always provides the most accurate and up-to-date information
Use chaing-of-thought to  plan and think about what are the queries that that can answer the user question.
your tools:search the web,get the current date

Use the following format of ReAct paradigm to how get the answer:
for N=2 times:
   
    Reason: you should always think and plan on the step you need to take: what are the best queries to search for the answer ?
    Action: choosing the actions to take and how to order them (it is recommended to send multple queries to cover more  and get best answer)
    Observation: Observing and reflecting on the received results of the actions, do they answer the question ?
    ... (this Reason/Action/Observation can repeat N times)
Final Thought: I now know the final answer,  generate the final answer based on retrieved information for the received query in free format text.

Hint:You can send mutliple queries to cover more results 
Final Answer: generate the final answer to the original question, very completely and  comprehensively to include all relevant information and details.
Your final answer does not need to be in the form of Thought/Action/Observation (that format is only for demonstrating how to
accomplish the task); simply generate the final answer based on the retrieved data. 
""")    

class OnlineRetrieverExpert:
    def __init__(self,model=_llm_model,cfg=cfg,repo_path:str=ospj(os.getcwd(), "output")):
        self.agent=Agent(model,  #choose_model()["experts"],  
                        name="OnlineRetrieverExpert",   
                        #output_type=SearchResults,
                        output_retries=cfg.workflow.no_retries,
                        system_prompt=web_search_agent_system_prompt,
                        output_type=str,
                        end_strategy="exhaustive",
                        tools=[Tool(self.online_search, takes_ctx=False), Tool(self.get_current_date, takes_ctx=False)]
                        )
        print(colored(f"OnlineRetrieverExpert Agent is ready and powerd by: {cfg.llm.experts.model}", "green"))
        self.error_it=0
        self.retrieved_information=""
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
            print(response)
            self.error_it=0
            #print(colored(f"Search results:\n{response}", "light_green"))
            return response
        except Exception as e:
            self.error_it+=1
            error_trace = traceback.format_exc()
            print(colored(f"Error during search: \n{error_trace}", "red"))
            return f"Error during search:\n {error_trace}\n if error persist more than 3 times (now it is {self.error_it}), it is recommended to use other search query or try again later."

    async def run(self, query: str, max_results: int = 10) -> str:
        """search online to for the query to get max of 10 results."""
        global total_tokens
        print(colored(f"Sending query to OnlineRetrieverExpert agent:\n{query}", "green"))
        response = await self.agent.run(f"The user query is:\n {query}.\nToday's date is {date.today()}. Use chain-of-thought to answer accurately and completel using online information\n")
        total_tokens = response.usage().total_tokens
        log_tokens(repo=ospj(os.getcwd(), "output"), agent_name=self.agent.name, llm_model=self.agent.model.model_name,
                   input_tokens=response.usage().input_tokens,
                   output_tokens=response.usage().output_tokens,
                   total_tokens=total_tokens)
        
            
        print(colored(f"\nTotal tokens used: {total_tokens}", "green"))
        return response.output
    
llm_settings = {
    "temperature": 0,


    "parallel_tool_calls ": True,
}
print(colored(f"Websearch Agent is ready and powerd by: {_llm_model}", "red"))
WebSearchAgent=Agent("openai:gpt-5-mini",  #choose_model()["expert"],  # OpenAIModel(_model_name_small, provider="openai") # OpenAIModel(_model_name_main_new, provider="openai") # OpenAIModel(_model_name_small_new, provider="openai") # OpenAIModel(_model_name_small, provider="openai")
                     name="OnlineRetrieverExpert",
                     system_prompt=web_search_agent_system_prompt,
                     end_strategy="exhaustive",  
                     
                     #output_type=SearchResults,
                     output_retries=10,
                     model_settings=llm_settings,
                     )
print(colored(f"Websearch Agent is ready and powerd by: {WebSearchAgent.model.model_name}", "green"))

@WebSearchAgent.tool_plain
def get_current_date(today:bool=True) -> str:
    """Ask and get the current (today) date to use for  up to-date search."""
    print(colored("Getting current date", "green"))
    return f"The  current date is {date.today()} \n"


@WebSearchAgent.tool_plain
def online_search(query: str, max_results: int = 10)-> str:
    """Search online to for the query to get max of 10 results."""
    print(colored(f"Searching for: {query}", "green"))
    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer="advanced"
        )
        print(colored(f"Search results: {response}", "light_green"))
        return response
    except Exception as e:
        error_trace = traceback.format_exc()
        print(colored(f"Error during search: \n{error_trace}", "red"))
        return f"Error during search:\n {error_trace}\n if error persist more than 3 times, it is recommended to use other search query or try again later."

async def agentic_web_search(query: str, max_results: int = 10) -> str:
    """Search online to for the query to get max of 10 results."""
    global total_tokens
    print(colored(f"Send query to WebRAG agent: {query}", "green"))
    response = await WebSearchAgent.run(f"The user query is:\n {query}.\nToday's date is {date.today()}. Use chain-of-thought to answer accurately and completel using online information\n")
    total_tokens = response.usage().total_tokens
    log_tokens(repo=ospj(os.getcwd(), "output"), agent_name=WebSearchAgent.name, llm_model=WebSearchAgent.model.model_name,
               input_tokens=response.usage().input_tokens,
               output_tokens=response.usage().output_tokens,
               total_tokens=total_tokens)
    
        
    print(colored(f"\nTotal tokens used: {total_tokens}", "green"))
    return response.output






async def chat_with_web_search_agent(user_input=List[str]):
    if not isinstance(user_input, list) and  isinstance(user_input, str):
        user_input = [user_input]
    elif isinstance(user_input, str):
        raise ValueError("user_input should be a list of strings or a single string")

    N=3
    assistant_reply = ""
    user_query = [""] * len(user_input)
    for i,question in enumerate(user_input):
        # Append the user input to the conversation history
        user_query[i]= f"Question{ i}:\n {question} \n"
       
        
        try:
            response=  WebSearchAgent( f"please provide a complete answer to this question using chain of thoughts:\n{question}. Note: the final answer should be very complete and comprehensive to include all relevant information and details.")
        except Exception as e:
            sys_prompt = f"""Answer the following question as best you can using the tools at your disposal.

            use your tools to search the web for the answer  or get the current date if needed.

            Use the following format:
            for N={N} times:
                Question: the input question you must answer 
                Thought: you should always think about what to do
                Action: the action to take
                Action Input: the input to the action which is the search query ( use chain of thought to generate the search query )
                Observation: the result of the action 
                ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: generate the final answer to the original question, very completely and  comprehensively to include all relevant information and details.
            The user question is:
            :\n ----- \n{question} \n------ \n

            """
            print(colored(f"\nError invoking WebSearchAgent from langchain: {e},\nWe use OpenAI Response instead.", "red"))
            print(colored(sys_prompt,'blue'))
            response = openai.responses.create(
                model="gpt-4o",
                tools=[{"type": "web_search_preview"}],
                input=sys_prompt,
                temperature=0, 
            )

            response = response.output[-1].content[0].text
        assistant_reply+= f"Assistant for Question {question}:\n {response} \n"

    return assistant_reply
if __name__ == "__main__":
    user_input = "what is the current reproduction number R_0 for measles in the US?"
    user_input="how to convert nx graph to sparse matrix csr in python using networkx?"
    agent=OnlineRetrieverExpert()
    async def main():
        response = await agent.online_search("where is mohammad hossein samaei living?")
        print(colored(response, "cyan"))

    import asyncio

    response=asyncio.run(main())

