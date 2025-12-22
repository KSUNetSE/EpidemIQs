from epidemiqs.agents.scientist_agent import BaseScientist
from typing import Any, Callable, Generic, Optional, Type, TypeVar, List,Dict
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
from os.path import join as ospj
import pandas as pd
import random

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
class ReviewResult:
    readability: str
    readability_score: int
    relevance: str
    relevance_score: int
    technical_soundness: str
    technical_soundness_score: int
    experimental_rigor: str
    experimental_rigor_score: int
    limitations: str
    limitations_score: int
    overall_review: str
    overall_score: float
    def to_dict(self):
        return self.__dict__.copy()
@dataclass
class ClassificationResult:
    objective_success: bool
    explanation: str    
    def to_dict(self):
        return self.__dict__.copy() 
class LLMAsJudgeAgent():
    def __init__(self, llm: str = None, cfg: Settings = None,rubrics: str = None,output_type:dataclass= ReviewResult, system_prompt: str = None):
        self.cfg = cfg if cfg is not None else get_settings()
        self.output_type=output_type if output_type is not None else ReviewResult
        self.reviewer_agent = BaseScientist[
        self.output_type,
        self.output_type
        ](
        llm=llm,
        cfg=cfg,
        name="LLM-as-Judge",
        phase="Review",
        system_prompt="You are an Editor in Chief of a prestigious journal. Please review the paper(s) carefully and score it based on the rubric that is provided to you according to how it could address the posed research question. Finally provide the final score and judgement. I if multiple papers are provided, please review them all and provide comparative scores and comments. " if system_prompt is None else system_prompt,
        scientist_output_type=self.output_type,
        phase_output_type=self.output_type,

        enable_plan=False,
        enable_reflect=False,
        enable_copilot=cfg.workflow.copilot,
        repo=os.path.join(os.getcwd(), "output"),
        ltm_path=None,
        ltm_cls=None,
        tools=[],
        default_tool=False,
        log_tokens_fn=log_tokens,
        agent_cls=Agent,
        )
        self.result="" 
        self.review_id=self._get_id()
        self.results_stored=True
        self.rubrics=rubrics if rubrics is not None else """
            1. Clarity & Writing Quality: Is the paper clearly written and well-structured? Are the ideas communicated effectively? Are details well mentioned and sections are comprehensive ?
                 Are sections logically organized and easy to follow ?\n
            2. Motivation & Relevance: Is the problem significant and well-motivated? Is it relevant to the question it was requested to address ? Is there any deviation from the main question ?\n

            3. Technical Soundness: Are the methods theoretically correct, well justified, and reproducible? Are assumptions reasonable? are results based on simulation results or beforementioned analytical results ?

            4. Experimental Rigor: Are experiments comprehensive, fair, and reproducible? Are baselines and metrics appropriate? Could they answer all aspect of the question ? 

            5. Limitations : Are limitations discussed and related to the work? Are ethical concerns or societal impacts appropriately addressed?
            range of Score: (0–10)
            Overall Score: please average all the above scores"""
    def _get_id(self):
        """Generates a unique 5-digit ID for the review."""
        return random.randint(0, 99999)
    def _reset(self):
        self.result="" 
        self.reviewer_agent._reset() 
    def save_review_content(self,review_data: Dict, output_dir=ospj(os.getcwd(), "output"),paper_path:str=None,question_id=None):
        """
        Saves the review dataclass to a .json file.
        
        Args:
            review_data (ReviewResult): The data class containing review details.
            output_dir (str): Folder to store json files.
        """
        # Ensure the ID is a string and 5 digits
        str_id = str(self.review_id).zfill(5)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        filename = f"{str_id}.json"
        filepath = os.path.join(output_dir, filename)
        review_data['paper_path']=paper_path
        review_data['question_id']=question_id
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                # Convert dataclass to dict and dump to json
                json.dump(review_data, f, indent=4)
            print(f"Saved JSON to: {filepath}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")

    def store_review_scores(self, readability_score:int, relevance_score:int, technical_soundness_score:int, experimental_rigor:int,limitations:int,overall_score:float=None,paper_path:str=None,question_id=None):
        """
        Appends review scores to an Excel file. Creates the file if it doesn't exist.
        
        Args:
            review_id (str|int): The 5-digit ID for the review.
            scores_dict (dict): Dictionary of metrics and scores (e.g., {'Clarity': 8, 'Tone': 9}).
            excel_file (str): Path to the Excel file.
        """
        excel_file="output/review_scores.xlsx"
        str_id = str(self.review_id).zfill(5)
        # Prepare the new row data
        new_data = {'Review_ID': str_id,
                    'Question_ID': question_id,
                    'Paper_Path': paper_path,
                    'Readability and Clarity': readability_score,
                    'Relevance': relevance_score,
                    'Technical Soundness': technical_soundness_score,
                    'Experimental Rigor': experimental_rigor,
                    'Limitations': limitations,
                    'overall_score': (readability_score + relevance_score + technical_soundness_score + experimental_rigor + limitations) / 5 if overall_score is None else overall_score
                    }
        
        new_row_df = pd.DataFrame([new_data])

        try:
            if os.path.exists(excel_file):
                # If file exists, read it to preserve existing data
                existing_df = pd.read_excel(excel_file)
                
                # Concat the new row to the existing dataframe
                updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            else:
                # If file doesn't exist, start a new dataframe
                updated_df = new_row_df
                print(f" Creating new Excel file: {excel_file}")

            # Save back to Excel
            updated_df.to_excel(excel_file, index=False)
            print(f"Appended scores for ID {str_id} to {excel_file}")
            self.results_stored=True
            return f"Scores successfully  stored!"
        except PermissionError:
            print(f" Error: Close {excel_file} before writing to it.")
            return f" Error: Close {excel_file} before writing to it."
        except Exception as e:
            print(f" Unexpected error writing to Excel: {e}")
            return f" Unexpected error writing to Excel: {e}"

    async def review_papers(self, papers: List[str], research_question: str, question_id, rubric: str = None) -> str:
        """
        Uses the LLM agent to review a list of papers based on a research question.
        
        Args:
            papers (List[str]): List of paper contents as strings.
            research_question (str): The research question to guide the review.
            
        
        Returns:
            str: The review summary and scores.
        """
        self.review_id=self._get_id()
        prompt = [f"Now please based on the following Research Question: {research_question}\n\nPlease review the following papers:\n\n" ]
        for paper in papers:
            with open(paper, "rb") as image_file:
                paper_data = image_file.read()
                file_name = os.path.basename(paper)

                prompt.append(f"The corresponding paper file named as:'{file_name}' submitted as\n")
                prompt.append(BinaryContent(data=paper_data, media_type='application/pdf' ))

        prompt.append(f"Based on the following rubric:\n{self.rubrics}.")

        print(colored(" Starting review process...", "cyan"))
        self.result = await self.reviewer_agent.forward(prompt)
        print(colored(" Review process completed.", "green"))

        # Save the full review text
        self.save_review_content(review_data=self.result.to_dict(), paper_path="-".join(papers), question_id=question_id)
        self.store_review_scores(
            readability_score=self.result.readability_score,
            relevance_score=self.result.relevance_score,
            technical_soundness_score=self.result.technical_soundness_score,
            experimental_rigor=self.result.experimental_rigor_score,        
            limitations=self.result.limitations_score,
            paper_path="-".join(papers)
            ,question_id=question_id
        )

                
        return self.result
class LLMAsClassifierAgent():
    """    
    Output: objective_success (bool)
    """

    def __init__(
        self,
        llm: str = None,
        cfg: Settings = None,
        output_type: dataclass = ClassificationResult,
        system_prompt: str = None
    ):
        self.cfg = cfg if cfg is not None else get_settings()
        self.output_type = output_type if output_type is not None else ClassificationResult

        self.classifier_agent = BaseScientist[
            self.output_type,
            self.output_type
        ](
            llm=llm,
            cfg=cfg,
            name="LLM-as-Classifier",
            phase="Classification",
            system_prompt=(
                "You are an Editor-in-Chief. Your only task is to determine whether a given paper "
                "correctly answers the provided research question.( you focus on correctness) "
                "Return ONLY a boolean field named 'objective_success': "
                "True if the paper correctly addresses the question; False otherwise if there are any mistakes, lack of clarity, or insufficient evidence. "
                "Use strict scientific judgment. if there are mistakes, lack of clarity, or insufficient evidence, return False. if the question addressed correctly, return True."
                if system_prompt is None else system_prompt
            ),
            scientist_output_type=self.output_type,
            phase_output_type=self.output_type,
            enable_plan=False,
            enable_reflect=False,
            enable_copilot=cfg.workflow.copilot,
            repo=os.path.join(os.getcwd(), "output"),
            ltm_path=None,
            ltm_cls=None,
            tools=[],
            default_tool=False,
            log_tokens_fn=log_tokens,
            agent_cls=Agent,
        )

        self.result = ""
        self.classification_id = self._get_id()
        self.results_stored = True

    def _get_id(self):
        return random.randint(0, 99999)

    def _reset(self):
        self.result = ""
        self.classifier_agent._reset()


    def save_classification(self, classification_data: Dict, output_dir=ospj(os.getcwd(), "output"),
                            paper_path: str = None, question_id=None):
        """
        Saves the classification boolean to a .json file.
        """
        str_id = str(self.classification_id).zfill(5)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        filename = f"{str_id}_classification.json"
        filepath = os.path.join(output_dir, filename)

        classification_data["paper_path"] = paper_path
        classification_data["question_id"] = question_id

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(classification_data, f, indent=4)
            print(f"Saved classification JSON to: {filepath}")
        except Exception as e:
            print(f"Error saving classification JSON: {e}")
    def store_classification_result(self, objective_success: bool, paper_path: str = None, question_id=None):
        """
        Stores the classification boolean result into an Excel file.
        Creates the file if it does not exist.
        
        Columns: 
        - Classification_ID
        - Question_ID
        - Paper_Path
        - Objective_Success (True/False)
        """

        excel_file = "output/classification_results.xlsx"
        str_id = str(self.classification_id).zfill(5)

        # Prepare new row
        new_data = {
            "Classification_ID": str_id,
            "Question_ID": question_id,
            "Paper_Path": paper_path,
            "Objective_Success": objective_success,
        }

        new_row_df = pd.DataFrame([new_data])

        try:
            if os.path.exists(excel_file):
                existing_df = pd.read_excel(excel_file)
                updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            else:
                updated_df = new_row_df
                print(f" Creating new Excel file: {excel_file}")

            updated_df.to_excel(excel_file, index=False)
            print(f"Appended classification result for ID {str_id} to {excel_file}")
            self.results_stored = True
            return "Classification result stored successfully!"

        except PermissionError:
            print(f" Error: Close {excel_file} before writing to it.")
            return f" Error: Close {excel_file} before writing to it."

        except Exception as e:
            print(f" Unexpected error writing to Excel: {e}")
            return f" Unexpected error writing to Excel: {e}"
    async def classify_papers(self, papers: List[str], research_question: str, question_id):
        """
        Returns True or False for each paper — whether it addresses the research question.
        """
        self.classification_id = self._get_id()


        prompt = [f"Research Question: {research_question}\n\n"
                  f"Determine whether each submitted paper answers this question.\n"]

        for paper in papers:
            with open(paper, "rb") as f:
                paper_data = f.read()
                file_name = os.path.basename(paper)

            prompt.append(f"Paper: '{file_name}' submitted as:\n")
            prompt.append(BinaryContent(data=paper_data, media_type='application/pdf'))


        print(colored("Starting classification...", "cyan"))
        self.result = await self.classifier_agent.forward(prompt)
        print(colored("Classification completed.", "green"))

        self.save_classification(
            classification_data=self.result.to_dict(),
            paper_path="-".join(papers),
            question_id=question_id
        )
        self.store_classification_result(
            objective_success=self.result.objective_success,
            paper_path="-".join(papers),
            question_id=question_id
        )

        return self.result

if __name__ == "__main__":

    import json
    from pathlib import Path
    
    def get_all_pdf_paths(directory_path):
        root_dir = Path(directory_path)
        pdf_paths = [str(file_path) for file_path in root_dir.rglob("*.pdf")]
        
        return pdf_paths


    def get_json_metadata(pdf_path):
        pdf_file = Path(pdf_path)
        

        json_path = pdf_file.parent / "question.json"
        
        if not json_path.exists():
            print(f"Warning: 'question.json' not found in {pdf_file.parent}")
            return None, None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                query_value = data.get("query")
                no_value = data.get("no")
                
                return query_value, no_value
                
        except json.JSONDecodeError:
            print(f"Error: 'question.json' in {pdf_file.parent} is not valid JSON.")
            return None, None

    from pathlib import Path
    #directory_path = "user-files/single-agent/4.1"
    directory_path = "output/epidemiqs"
    all_reports= get_all_pdf_paths(directory_path)
    async def main_reviewer():
        cfg = get_settings()
        reviewer = LLMAsJudgeAgent(llm="gpt-4o", cfg=cfg)
        for paper in all_reports:
            print(f"Reviewing paper: {paper}")
            papers = [paper]
            get_json_metadata(paper)
            question, question_id = get_json_metadata(paper)
            print(f"Question ID: {question_id}")
            review = await reviewer.review_papers(papers, research_question=question, question_id=question_id)
            reviewer._reset()
            print("Final Review:\n", review)
    async def main_classifier():
        cfg = get_settings()
        classifier = LLMAsClassifierAgent(llm="gpt-5-mini", cfg=cfg)
        for paper in all_reports:
            print(f"Classifying paper: {paper}")
            papers = [paper]
            question, question_id = get_json_metadata(paper)
            if str(question_id) == str(5):
                question+="Correct answer for random vaccination is 75 percent and for targetted is 11.25 or 9/80 percent(10 percent is also acceptable), and simulate the result. "
            elif str(question_id) == str(4):
                question+="Correct answer should exhibit coexisitence of two strains with correct parameters, and validate with stochastic simulations. "
            elif str(question_id) == str(3):
                question+="Correct answer should analyze the difference between spread over temporal and static networks, and validate with stochastic simulations and data analysis of results "
            elif str(question_id) == str(2):
                question+="Correct answer should analyze the the both tranmission break reasons and show both are can lead to break of transmission, and validate with stochastic simulations and data analysis of results "
            elif str(question_id) == str(1):
                question+="Correct answer should correcttly analyze the effect of degree-heterogeneous networks on disease dynamics compared to homogeneous-mixing networks, and validate with stochastic simulations and data analysis of results over two network types to show how it differes from homogeneous-mixing networks. "
            print(f"Question ID: {question_id}")
            classification = await classifier.classify_papers(papers, research_question=question, question_id=question_id)
            classifier._reset()
            print("Final Classification:\n", classification)
    asyncio.run(main_reviewer())
    #asyncio.run(main_classifier())
