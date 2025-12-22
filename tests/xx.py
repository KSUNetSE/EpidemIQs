from pydantic_ai import Agent
import dotenv
from epidemiqs.utils.llm_models import choose_model
from epidemiqs.agents.discovery import DiscoveryScientist
from pydantic_ai.tools import Tool
from termcolor import colored
from pydantic.dataclasses import dataclass
import asyncio
import os
@dataclass
class SecretaryAgent:
    """Data structure to hold the secretary agent's decision."""
    in_scope_of_EpidemIQs: bool
def talk_to_user(message:str):
        """through this function you can communicate with the user directly"""
        print(colored(message,"magenta"))
        print(colored("--------------------------------------------------","magenta"))
        #user_response = input(colored(f"{message} \n ------------------- \n ", "green"))
       #return user_response
    
secretary_agent=Agent(
            name="SecretaryAgent",
            model=choose_model()["expert"],
            system_prompt=r"""You are a meticulous and professional secretary agent, responsible for classifying the user query to decide wether it is in the scope of network based epidemic modeling or not. wrong classification may lead to failure of the whole simulation task. If the user query is related to network based epidemic modeling, you should mark EpidemIQs:True so that the query is sent to the framwork for epidemic modeling. If it is not related to network based epidemic modeling, you should mark EpidemIQs:False; if it is not related to EpimemIQs scope, you can provide general assistance to user through tool " talk_to_user ". You should always provide the final answer in strucuture output as:\n
            EpidemIQs:Boolean value(True/False)\n
            Reasoning: your reasoning about the classification\n
            this answer will be the switch to trigger EpidemIQs framework or not. The user will not see the final output, so make sure to communicate clearly through the tool "talk_to_user" and provide necessary information to user and explain either or not EpidemIQs framework can help them.If it is not in scope,  enure to answer their query wiht your own knwoledge comprehensively. if it is in scope, explain to user that their query will be processed by EpidemIQs framework.""",
            tools=[Tool(talk_to_user, takes_ctx=False)],
            output_retries=5,
            output_type=SecretaryAgent,
            end_strategy="exhaustive",
            
            )

if __name__ == "__main__":
    async def run_secretary_agent( user_query="can I use EpidemIQs"):
        response =await  secretary_agent.run(user_prompt=user_query)
        print(colored(f"Secretary Agent Response: {response}","blue"))

    asyncio.run(run_secretary_agent())
