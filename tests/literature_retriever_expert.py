import json
import os
import requests
from time import sleep
from os.path import join as ospj
from termcolor import colored
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext, ModelRetry
from typing import List, Dict, Optional, Union
import traceback
from utils import log_tokens
from datetime import date
from llm_models import choose_model
from pydantic_ai.tools import Tool

import logfire
from termcolor import colored, cprint
load_dotenv()

semantic_api_key=os.getenv('semantic_api_key')
_llm_model = choose_model()["expert"]  # Default to the expert model, can be changed to "math" or "scientist"
#print(colored(f"SemanticScholarTool Agent is ready and powered By: {_llm_model}","red"))

literature_review_agent_prompt="""
You are a Ph.D. level smart agent  with sharp and accurate mind that extract the most meaningful information from literuature, who looks through papers on a specific topic, summarize them and give final results.You are as assistant to Discovery Scientist to retreive literature reviews.\n
The results should be written scientifically and professionally. containing important infomration with references for the given papers.\n
As a smart agent, use self reflecting and chain of thoughts to extract most meaningul information from the given papers\n
also provide answers based on the acquired information, if you are asked any question. \n
Important: your answer should be based on the information you have acquired from the papers and reflecting on them, if not enough information is available, you should say "I can not answer this question based on the available information, please ask another question or provide more information".\n
"""
literature_review_agent_prompt ="""You are a Ph.D. level smart agent who sharp and accurate in extracting the most meaningful, relevant, and accurate information from literature, who looks through papers on a specific topic, summarizes them to represent the findings with details regarding the query.\n
The results should be presented in a scientific and professional manner, containing important information with relevant references to supporting papers.\n
As a smart agent, use self-reflecting and chain of thoughts in extracting the most meaningful  and relevant information from the given papers according to the requested query.\n
Also be available to provide answers based on the acquired information, if you are asked any question.\n
IMPORTANT: your answer should be based on the information you have acquired from the papers; if not enough information is available, you should say ``I can not answer this question based on the available information for the requested query, please ask another question or suggest another query''\n
generate the final answer to the original question(query), completely and comprehensively to include all relevant information and details, including citations (but NEVER include bibliography in your respone (it is waste of tokens)) just cite the relevant work in your answer (using bibitem format) I already included the bibliography.\n
IMPORTANT: While your answer should be comprehensive, DO NOT include irrelevant references in your response.

Please perform the ReAct(Reason-Action) paradigm to generate your response as:\\
for N=maximum 2 times per query:

    Thought: you should always think and plan on the step you need to take
    to search for the answer?
     Action: choosing the actions (searching for suitable query) to take and how to order them (You can send maximum three request with 
    different queries to search for the query, it is recommended that to do it sequentially, if the first request does 
    not return satisfactory results, you can retry with  different topic)
     Observation: Observing and reflecting on the received results of the actions, do they answer the question? are they relevant and sufficient to answer the question?
    ... (this Thought/Action/Action Input/Observation can repeat N times)
Final Thought: I now know the final answer based on the retrieved data and I generate my final asnwer.\n
}
your final answer does not need to be in the form of Thought/Action/Observation (that format is only for showing how to accomplish the task), just generate the final answer based on the retrieved data.\n

Hint: You can send multiple queries to cover more results.\\
\textbf{Final Answer:} generate the final answer to the original question, completely and comprehensively to include all relevant information and details, including citations (but no bibliography is needed, just cite the relevant work in your answer (using \texttt{bibitem} format) I already included the bibliography. \\
"""

@dataclass
class literature_review:
    title: str
    abstract: str
    summary:str



class LiteratureReviewAgent:
    def __init__(self, semantic_api_key=os.getenv('semantic_api_key'), json_file=ospj("output","literature_review.json"), llm_sys_prompt=None, cfg=get_settings()):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {"X-API-KEY": semantic_api_key} if semantic_api_key else {}
        self.json_file = json_file
        print(colored(f"SemanticScholar API Key: {semantic_api_key}", "blue"))
        self.llm_sys_prompt = llm_sys_prompt
        self.results = self._load_json()  
        self.queries=[]
        self.literature_review_agent = Agent(
            model=_llm_model,
            name="LiteratureRetrieverExpert",
            system_prompt=self.llm_sys_prompt,
            result_tool_name="final_result",
            result_retries=5,
            retries=10,
            result_tool_description=str,
            end_strategy="exhaustive",
            tools=[Tool(self.conduct_review, takes_ctx=False)],

        )
        self.memory=None
        print(colored(f"LiteratureReview Agent is ready and powered By: {self.literature_review_agent.model.model_name}","blue"))

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

    def _search_papers(self, query, limit=7):
        """Search for papers based on a query using the Semantic Scholar API."""
        endpoint = f"{self.base_url}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,abstract,url,citationCount"
        }
        response = requests.get(endpoint, params=params, headers=self.headers)
        sleep(1)
        response.raise_for_status()
        print(colored(f"Papers: {response}", "red"))
        print(colored(f"\n{"-"*25}\n", "red"))
        print(colored(f"Response: {response.json()}", "red"))


        return response.json()

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
        print(colored(f"Papers: {data}", "green"))
        return data

    async def conduct_review(self, query:str):#,  summary=False):
        """
        Return found papers for a given query from Semantic Scholar API.
        """
        summary=False
        print(colored(f"Searching for: {query} and  request for summary: {summary}", "blue"))
        
        paper_count=7
        if self.queries!= []:
            for existing_query in self.queries:
                if existing_query.lower() != query.lower():
                    self.queries.append(query.lower())
                    
        else:
            self.queries.append(query.lower()) 
                  
        try:
            existing_entry = next((entry for entry in self.results["results"] if entry["query"] == query), None)

            if existing_entry:
                print(colored(f"Found existing papers for\n {query}\n, we cancel the search.", "green"))
                print(colored(f"Existing papers\n: {existing_entry}\n", "green"))
                #self.results["results"].append(existing_entry)  # Append the existing entry to results
                if summary:
                    return  await self.summarize_results(query=query)
                else:
                    
                    return existing_entry

            else:
                papers = self._search_papers(query, limit=paper_count)

                if not papers.get("total"):
                    print(colored(f"No papers found for query: {query}", "red"))
                    return f"No papers found for query: {query}"

                query_data = {"query": query, "papers": [], "summary": ""}
                seen_paper_ids = set()  # Avoid duplicates
                for paper in papers.get("data", []):
                    paper_id = paper.get("paperId")
                    if not paper_id or paper_id in seen_paper_ids:
                        continue
                    seen_paper_ids.add(paper_id)

                    details = self._get_paper_details(paper_id)
                    if not details:
                            continue

                    details = self._get_paper_details(paper["paperId"])
                    authors = details.get('authors', [{'name': 'Unknown'}])
                    journal_info = details.get('journal') or {}  # Handle None journal
                    first_author = authors[0]['name'] if authors else 'Unknown'
                    citation_info = {
                    "authors": ", ".join([author.get('name', 'Unknown') for author in authors[:3]]) +
                               ("..." if len(authors) > 3 else ""),
                    "year": details.get('year', 'Unknown'),
                    "title": details.get('title', 'Unknown'),
                    "journal": journal_info.get('name', 'Unknown') if journal_info else 'Unknown',
                    "volume": journal_info.get('volume', '') if journal_info else '',
                    "pages": journal_info.get('pages', '') if journal_info else '',
                    "doi": details.get('externalIds', {}).get('DOI', 'Unknown'),
                    "url": f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None
                        }
                    formatted_citation = f"{citation_info['authors']} ({citation_info['year']}). {citation_info['title']}. "
                    if citation_info['journal'] != 'Unknown':
                        formatted_citation += f"{citation_info['journal']}"
                        if citation_info['volume']:
                            formatted_citation += f", {citation_info['volume']}"
                        if citation_info['pages']:
                            formatted_citation += f", {citation_info['pages']}"
                    formatted_citation += f". DOI: {citation_info['doi']}"                   
                    query_data["papers"].append({
                        "paper_title": details.get("title", "Unknown"),
                        "abstract": details.get("abstract", "No abstract available"),
                        "citation": formatted_citation,
                        "summary (tldr)": details.get("tldr", "No summary available")
                    })
                
                self.results["results"].append(query_data)
                self._save_json()  
                print(colored(f"Saved {len(query_data['papers'])} papers for '{query}'.", "green"))

            if summary:
                 return  await self.summarize_results(query=query)
            else:
                return query_data
        except Exception as e:
                error_trace=traceback.print_exc()
                print(colored(f"Error conducting review: {error_trace}", "red"))
                return f'Error conducting review: {e}, if the eror is 400, Bad Request, it means the query is not found in the database, use more general query, remember the search is Semantic Scholar and  you are searching for articles related.'
                

    async def summarize_results(self,query):
        """agent to retrieve and summarize stored literature review results."""
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
        summary = await self.literature_review_agent.run(text_to_summarize, message_history=self.memory)
        output_dir = os.path.join(os.getcwd(),"output")
        log_tokens(repo=output_dir, agent_name=self.literature_review_agent.name, llm_model=self.literature_review_agent.model.model_name,
                   request_tokens=summary.usage().request_tokens,
                   response_tokens=summary.usage().response_tokens,
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
                
                
        
        response = await self.literature_review_agent.run(prompt, message_history=self.memory)
        self.memory=response.all_messages()    
        return response.output  

    async def literature_review(self,query: str, prompt: Optional[str] = None):
        """
        Conduct a literature review by searching for papers and adding them to stored results.
        If summary=True, the LLM agent will summarize the stored papers.
        """
        lit_prompt=f"Please search for the following quer\n{query} and provide the results in a comprehensive way"
        if prompt is not None:
            lit_prompt+=f"Also consider the Disc hvvvovery Scinetist prompt\n{prompt}"
        response = await self.literature_review_agent.run(user_prompt=lit_prompt, summary=self.memory)
        self.memory=response.all_messages()
        return response.output
if __name__ == "__main__":
    import asyncio
    async def main():

        # Initialize the agent with your API key
        #print(colored(f"SemanticScholar API Key: {semantic_api_key}", "blue"))
        agent = LiteratureReviewAgent(semantic_api_key=semantic_api_key,json_file="output/XXXXXXX.json")

        #Conduct reviews for multiple queries
        #result= await agent.conduct_review(query="HIV in Iran", summary=True)
        #result=agent._search_papers(query="Dengue model prediction compartmental model")
        result=await agent.literature_review(query="", prompt="what charateristics of multi layer network topologies enable coexistence of exclusive bi-virus spread?")
        print(colored(f"Result: {result}", "cyan"))
        print(colored(f"Queries: {agent.queries}\n ----------------\n", "magenta"))
        print("*"*50+ "\n")
        print(colored(f"Results:\n {result}\n----------------\n", "magenta"))
        print(colored("*"*50, "magenta"))
        print(colored(f"Memory:\n {agent.memory}\n----------------\n", "green"))
        print(colored("*"*50, "magenta"))
        result2=await agent.talk_to_literature_review_agent(prompt="so what is the most imortant characterisitc??")
        print(colored(f"Result2: {result2}", "cyan"))
        print(colored("*"*50, "magenta"  ))
        print(colored(f"Queries: {agent.memory}\n ----------------\n", "green"))
        print("*"*50+ "\n")
    asyncio.run(main())
