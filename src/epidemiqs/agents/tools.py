# tools
import pandas as pd
import numpy as np
from os.path import join as ospj
import scipy, math, statistics,  datetime, time, json,csv, os, sys
import re
import sympy as sp
import networkx as nx
from scipy import sparse
import fastgemf as fg
from termcolor import colored, cprint
from typing import List, Dict, Any, Optional, Union
import traceback
import requests
from time import sleep
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext, ModelRetry
from datetime import date
from epidemiqs.utils.llm_models import choose_model
import logfire
from tavily import TavilyClient
from contextlib import contextmanager
import textwrap
import signal
from epidemiqs.utils.config import get_settings, Settings

from epidemiqs.utils.utils import log_tokens

client= TavilyClient(os.environ["TAVILY_API_KEY"])
load_dotenv()


class TimeoutException(Exception):
    """Custom exception for timeout"""
    pass

class CodeExecutor:
    def __init__(self, default_timeout: int = 60):
        # Initialize the persistent environment with standard libraries
        self.globals = {
            "__builtins__": __builtins__,
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
            "scipy": __import__("scipy"),
            "math": __import__("math"),
            "statistics": __import__("statistics"),
            "datetime": __import__("datetime"),
            "time": __import__("time"),
            "json": __import__("json"),
            "csv": __import__("csv"),
            "os": __import__("os"),
            "sys": __import__("sys"),
            "re": __import__("re"),
            "nx": __import__("networkx"),
            "sparse": __import__("scipy.sparse"),
            "sp": __import__("sympy"),
            "sklearn": __import__("sklearn"),
            "fg": __import__("fastgemf"),
            
        }
        self.code_history = {"": ""}
        self.default_timeout = default_timeout

    @contextmanager
    def timeout_context(self, seconds: int):
        """Context manager for timeout handling"""
        def timeout_handler(signum, frame):
            raise TimeoutException(f"Code execution exceeded timeout of {seconds} seconds")
        
        # Set the signal handler and alarm
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    async def execute(
        self, 
        code: str, 
        return_vars: List[str] = None, 
        write_to_file: bool = True, 
        script_name: str = "python_code.py",
        timeout: int = None
    ) -> str:
        """
        Execute Python code with timeout protection.
        
        Args:
            code: The Python code to execute.
            return_vars: List of variable names to return after execution.
            write_to_file: Whether to save the accumulated code to a file.
            script_name: File name for saving the code.
            timeout: Maximum execution time in seconds (uses default_timeout if None).
        
        Returns:
            A string message indicating success or failure, including results.
        """
        if timeout is None:
            timeout = self.default_timeout
            
        path = None
        cwd = os.getcwd()
        if script_name:
            path = os.path.join(cwd, "output", os.path.basename(script_name))
            os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            print(colored( f"\nExecuting code with timeout={timeout}s...\n","blue"))
            print(colored(f"Params: {return_vars}","blue"))
            print(colored(f"Write to file: {write_to_file}","blue"))
            print(colored(f"Path: {path if path else script_name}","blue"))
            print(colored(f"Code:\n{code}","blue")      )
        except:
            pass  #  colored is not available

        try:
            with self.timeout_context(timeout):
                #exec(textwrap.dedent(code), self.globals) # dedent to fix indentation issues, but may cause problems with certain code structures, especially when multiple function or classes are defined and used recursively.
                exec(code, self.globals)
            
            # add to code history
            if script_name not in self.code_history:
                self.code_history[script_name] = ""
            self.code_history[script_name] += "\n" + code

            # prepare results
            message = f"Code executed under {timeout}s."
            if return_vars:
                result = {var: self.globals[var] for var in return_vars if var in self.globals}
                message += f"\nReturned Variables: {result}\n"

            # write to file if requested
            if write_to_file:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.code_history[script_name])
                message += f" and saved to {path}"

            return message

        except TimeoutException as e:
            error_msg = (
                f"TIMEOUT ERROR: {str(e)}\n\n"
                f" SUGGESTIONS TO REDUCE TIME COMPLEXITY:\n"
                f"1. Reduce input size or use sampling for large datasets\n"
                f"2. Use vectorized operations (NumPy/Pandas) instead of loops\n"
                f"3. Optimize algorithms (if applicable)\n"
                f"4. Profile the code to identify bottlenecks\n"
                f"5. Break down the task into smaller, manageable chunks\n"
                f"6. Use more efficient data structures (e.g., sets for lookups)\n"
                f"Current timeout: {timeout}s. Consider optimization before increasing."
            )
            print(f"Timeout error:\n{error_msg}")
            return error_msg

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error executing code:\n{error_trace}")
            return (
                f"Error executing code:\n{error_trace}\n"
                f"Observe the return error through chain of thought, and if ran into persistent errors, "
                f"try to accomplish task in simpler and more manageable code blocks or start from scratch using other ways\n\n"
            )

    def reset_environment(self):
        """Reset the execution environment and code history"""
        self.__init__(default_timeout=self.default_timeout)

    def set_timeout(self, timeout: int):
        """Update the default timeout"""
        self.default_timeout = timeout
        
        

class LiteratureReview:
    def __init__(self,
                 cfg:Settings=None,
                 json_file=ospj(os.getcwd(),"output","literature_review.json"), 
                 sort_type:str="influentialCitationCount"):
        self.cfg=cfg or get_settings()
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = { "X-API-KEY": self.cfg.apis.SEMANTIC_API_KEY.get_secret_value() if self.cfg.apis.SEMANTIC_API_KEY else os.getenv("SEMANTIC_API_KEY") }
        self.json_file = json_file
        self.results = self._load_json()  
        self.queries=[]
        self.sort_type=sort_type
        self.no_paper_per_query=7
        self.max_retrieved_allowed=50
        self.no_retrieved_papers=0

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
                f"(sorted by recency)", "green"
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
    
if __name__ == "__main__":
    # Example usage
    import asyncio
    async def main():
        a = LiteratureReview(sort_type="recency")
        result = await a.conduct_review(query="competitive SIS epidemic spreading models multiplex networks coexistence dominance network structure influence")
        return result
    result=asyncio.run(main())
    print(result)