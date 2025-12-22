import os
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.grok import GrokProvider

from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai import Agent

from epidemiqs.config import get_settings, Settings
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()


def choose_model(cfg:Settings=None):
    cfg=get_settings() if cfg is None else cfg
    #print(cfg)
    llms_models={}
    for role in ["scientists", "experts", "mathematician_expert", "vision_expert"]:
        role_cfg = getattr(cfg.llm, role)
        
        try:
            if role_cfg.provider.lower() == "openai":
                model = OpenAIChatModel(role_cfg.model, provider=OpenAIProvider(api_key=cfg.apis.OPENAI_API_KEY.get_secret_value() if cfg.apis.OPENAI_API_KEY else None))
                llms_models[role] = model
                
            elif role_cfg.provider.lower() == "xai" or role_cfg.provider.lower() == "grok":
                model = OpenAIChatModel(role_cfg.model, provider=GrokProvider(api_key=cfg.apis.XAI_API_KEY.get_secret_value() if cfg.apis.XAI_API_KEY else None))
                llms_models[role] = model

            elif role_cfg.provider.lower() == "gemini" or role_cfg.provider.lower() == "google":
                #print(f"Configuring LLM for role: {role} with provider: {role_cfg.provider} and model: {role_cfg.model}")
                provider = GoogleProvider(api_key=cfg.apis.GEMINI_API_KEY.get_secret_value() if cfg.apis.GEMINI_API_KEY else None)
                model = GoogleModel(role_cfg.model, provider=provider)
                llms_models[role] = model
                
            elif role_cfg.provider.lower() == "anthropic":
                model = AnthropicModel(role_cfg.model, provider=AnthropicProvider(api_key=cfg.apis.ANTHROPIC_API_KEY.get_secret_value() if cfg.apis.ANTHROPIC_API_KEY else None))
                llms_models[role] = model
            else:
                raise ValueError(f"Unsupported provider: {role_cfg.provider} for role: {role}")
        except Exception as e:
            raise ValueError(f"Error initializing model for role {role}: {e}")
    #print(llms_models)
    return llms_models    
            # You can add more providers here as needed, please check the pydantic-ai documentation for supported providers and models. (https://ai.pydantic.dev/models/overview/)


if __name__ == "__main__": 
    # Example usage of the Agent class
    from pydantic_ai import Agent
    from termcolor import colored as colored
    import asyncio
    @dataclass
    class answer:
        answer: str
        reasoning: str
        date: str
    async def test_agent(): 
        cfg=get_settings()
        print(cfg)
        agent2=Agent(model=choose_model(cfg)["scientists"],system_prompt="you are a good agent who is exteremely passionate and emotional and very talkative")
        agent1=Agent(model=choose_model(cfg)["experts"],system_prompt="you are a good agent who is exteremely passionate and emotional and very talkative")
        agent3=Agent(model=choose_model(cfg)["vision_expert"],system_prompt="you are a good agent who is exteremely geek in math and very talkative")
        print(agent1.model.model_name,agent2.model.model_name,agent3.model.model_name)
        response=await agent1.run(user_prompt="what company owns you ?",output_type=answer)
        print(colored(response.output,"green"))
        response=await agent2.run(user_prompt="what company owns you ?",output_type=answer)
        print(colored(response.output,"red"))
        response=await agent3.run(user_prompt="what company owns you ?",output_type=answer)
        print(colored(response.output,"blue"))
    asyncio.run(test_agent())