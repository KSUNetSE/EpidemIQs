import asyncio
import json
import functools
import os,csv
from os.path import join as ospj
from datetime import datetime
from pydantic_ai import RunContext
import shutil

from pathlib import Path
import os
from collections import defaultdict
def tool_logger(save_path, file_name):
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Capture input
            input_data = {
                "args": args,
                "kwargs": kwargs
            }
            
            # Execute async function
            output = await func(*args, **kwargs)
            
            # Skip saving if output starts with "error"
            if isinstance(output, str) and output.lower().startswith("error"):
                print(f"Skipping saving for {func.__name__} due to error in output.")
                return output
            
            else:
                call_record = {
                    "name": func.__name__,
                    "input": input_data,
                    "output": output
                }
                
                # Ensure save directory exists
                os.makedirs(save_path, exist_ok=True)
                file_path = os.path.join(save_path, file_name)
                
                # Read existing data or initialize
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        try:
                            data = json.load(f)
                            if not isinstance(data, dict) or "tool_calls" not in data:
                                data = {"tool_calls": []}
                        except json.JSONDecodeError:
                            data = {"tool_calls": []}
                else:
                    data = {"tool_calls": []}
                
                # Append new call to tool_calls list
                data["tool_calls"].append(call_record)
                
                # Write updated data back to file
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                return output
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Capture input
            input_data = {
                "args": args,
                "kwargs": kwargs
            }
            
            # Execute sync function
            output = func(*args, **kwargs)
            
            # Skip saving if output starts with "error"
            if isinstance(output, str) and output.lower().startswith("error"):
                print(f"Error: {output}")
                return output
            
            else:
                call_record = {
                    "name": func.__name__,
                    "input": input_data,
                    "output": [output]
                }
                
                # Ensure save directory exists
                os.makedirs(save_path, exist_ok=True)
                file_path = os.path.join(save_path, file_name)
                
                # Read existing data or initialize
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        try:
                            data = json.load(f)
                            if not isinstance(data, dict) or "tool_calls" not in data:
                                data = {"tool_calls": []}
                        except json.JSONDecodeError:
                            data = {"tool_calls": []}
                else:
                    data = {"tool_calls": []}
                
                # Append new call to tool_calls list
                data["tool_calls"].append(call_record)
                
                # Write updated data back to file
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                return output
            
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

@tool_logger(save_path="output", file_name="tool_call.json")
def add_numbers(a, b, c=0):
    try:
       with open("output/tool_call.json", "r") as file:
            data = json.load(file)
            if not isinstance(data, dict) or "tool_calls" not in data:
                data = {"tool_calls": []}
    except Exception as e:
        return f"Error: {str(e)}"

@tool_logger(save_path="output", file_name="tool_call.json")
def multiply_numbers(a, b):
    return a * b

@tool_logger(save_path="output", file_name="tool_call.json")
async def async_multiply_numbers(a, b):
    await asyncio.sleep(1)  # Simulate async behavior
    return a * b

def get_user_prompt_from_runcontext(run_context: RunContext) -> list[str]:
    user_prompts = []
    for message in run_context.messages:
        for part in message.parts:
            if part.part_kind == 'user-prompt':
                user_prompts.append(part.content)
    return user_prompts


def find_files_repo(repo_path:str, extension:str,full_path_bool=False):
    """
    Returns a list of full paths to files with the specified extension in the repo_path.

    Parameters:
    - directory (str): The directory to search.
    - extension (str): The file extension to filter by (e.g., '.txt').

    Returns:
    - List of file names with that extension.
    """
    matching_files = []
    for entry in os.listdir(repo_path):
        full_path = os.path.join(repo_path, entry)
        if os.path.isfile(full_path) and entry.lower().endswith(extension.lower()):
            if  full_path_bool:
                matching_files.append(full_path)
            else:
                matching_files.append(entry)
    return matching_files

def log_tokens(repo=None,agent_name="NA",llm_model="NA", input_tokens=0, output_tokens=0, total_tokens=0,csv_name:str="",time:float=0.0):
    if repo is None:
        repo=ospj(os.getcwd(), "output")
    if not os.path.exists(repo):
        os.makedirs(repo)
    path=ospj(repo, "token.csv") if csv_name=="" else ospj(repo, csv_name)
    print(f"Logging tokens to {path}")
    write_header = not os.path.exists(path)
    with open(path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["timestamp", "Agent-Name", "LLM", "input_tokens", "output_tokens", "total_tokens","time"])
        writer.writerow([datetime.now(), agent_name, llm_model,input_tokens, output_tokens, total_tokens,time])
        
def count_pdfs_by_folder(root: Path | str = Path("/")) -> dict[str, int]:
    """
    Walk through `root` and return a mapping:
        {absolute_folder_path: number_of_pdf_files_in_that_folder_and_descendants}
    """
    root = Path(root).expanduser().resolve()
    counts: defaultdict[str, int] = defaultdict(int)

    for path in root.rglob("*.pdf"):
        # path.parent is the directory containing this PDF;
        # path.parents includes all ancestor directories up to `root`
        for parent in path.parents:
            if parent == root.parent:  # stop once we move above the chosen root
                break
            counts[str(parent)] += 1

    return dict(counts)
       
       
       
def move_to_destination(source_dir,destination):
    os.makedirs(destination, exist_ok=True)
    # move all files in source_dir to destination
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        if os.path.isfile(source_file):  # Ignore subfolders
            shutil.move(source_file, os.path.join(destination, filename))
# Or: repo_directory = 'C:/phd/GEMF_LLM/output'
if __name__ == "__main__":
    repo_directory = os.path.join(os.getcwd(), "output", "singleagent", "o3")
    png_files = find_files_repo(repo_directory, '.py',True)
    print(png_files)
    from pydantic_ai import Agent, RunContext
    #agent= Agent("gpt-4.1", name="test_agent", system_prompt="answer everything in finglish")
    #print(f"Agent Name: {agent.name}, Model: {agent.model.model_name}")
    #result = agent.run_sync("how are you?")
    #print(f"Result: {result}")
    log_tokens(repo=repo_directory,csv_name="kossher.csv" ,agent_name="test_agent", llm_model="gpt-4.1", input_tokens=100, output_tokens=200, total_tokens=300)
    # Example usage:
    #repo_directory = 'C:/phd/GEMF_LLM/output'
    #png_files = find_files_repo(repo_directory, '.png',True)
    #print(png_files)
    #txt_files = find_files_repo(repo_directory, '.txt') # Example for another extension
    #print(txt_files)
   # print(find_files_repo(os.path.join(os.getcwd(), "output") , ".png"))

