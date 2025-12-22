from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, SecretStr, ValidationError, PositiveInt, NonNegativeInt
from pydantic_settings import BaseSettings, SettingsConfigDict
from termcolor import colored
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv()

ProviderName = Literal["openai", "grok", "xai", "google", "gemini", "anthropic"]

# Workflow Config
class WorkflowModules(BaseModel):
    plan: bool = True
    reflect: bool = True


class WorkflowCfg(BaseModel):
    copilot: bool = False
    scientist_modules: WorkflowModules = WorkflowModules()
    reflection_max_iters: NonNegativeInt = 1
    no_retries: PositiveInt = 5
    no_paper_revise: NonNegativeInt = 0
    time_out_tools: PositiveInt = 180  # default timeout for tool execution in seconds


# LLM Config 
class LLMRoleCfg(BaseModel):
    provider: ProviderName = "openai"
    model: str = "gpt-4.1"


class LLMCfg(BaseModel):
    scientists: LLMRoleCfg = LLMRoleCfg(provider="openai", model="gpt-4.1")
    experts: LLMRoleCfg = LLMRoleCfg(provider="openai", model="gpt-4.1-mini")
    mathematician_expert: LLMRoleCfg = LLMRoleCfg(provider="openai", model="o3-mini")
    vision_expert: LLMRoleCfg = LLMRoleCfg(provider="openai", model="gpt-4.1-mini")


# API keys
class APIs(BaseSettings):
    OPENAI_API_KEY: Optional[SecretStr] = None
    GEMINI_API_KEY: Optional[SecretStr] = None  
    XAI_API_KEY: Optional[SecretStr] = None        # Grok
    ANTHROPIC_API_KEY: Optional[SecretStr] = None
    TAVILY_API_KEY: Optional[SecretStr] = None
    SEMANTIC_API_KEY: Optional[SecretStr] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# paths
class PathsCfg(BaseModel):
    output_dir: str = "outputs"
    contact_network_path: Optional[str] = None
    data_path: Optional[str] = None


class Settings(BaseSettings):
    query: str = "your query here"
    name: str = "experiment1"
    workflow: WorkflowCfg = WorkflowCfg()
    llm: LLMCfg = LLMCfg()
    paths: PathsCfg = PathsCfg()
    apis: APIs = APIs()

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Settings":
        import yaml
        data = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            print(colored(f"[Warning] Config file {path} not found, using defaults and .env vars.", "red"))
        cfg = cls(**data)
        #print(colored(cfg, "cyan"))
        return cfg

    def validate_provider_secrets(self) -> None:
        """Validate provider API keys based on the active providers and warn for optional ones."""

        provider_to_key = {
            "openai": "OPENAI_API_KEY",
            "grok": "XAI_API_KEY",    
            "xai": "XAI_API_KEY",  
            "google": "GEMINI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            
        }

        roles = {
            "scientists": self.llm.scientists,
            "experts": self.llm.experts,
            "mathematician_expert": self.llm.mathematician_expert,
            "vision_expert": self.llm.vision_expert,
        }

        missing = []

        for role_name, role_cfg in roles.items():
            provider = role_cfg.provider
            key_name = provider_to_key.get(provider)

            if key_name is None:
                print(colored(f"[Warning] {role_name}: unknown provider '{provider}'", "yellow"))
                continue

            key_value = getattr(self.apis, key_name, None)
            if key_value is None or (
                isinstance(key_value, SecretStr) and not key_value.get_secret_value()
            ):
                env_val = os.getenv(key_name)
                if not env_val:
                    missing.append((role_name, provider, key_name))

        if missing:
            err_msg = "\n".join(
                f" - Role '{r}' uses provider '{p}' but {k} is missing in .env or environment."
                for r, p, k in missing
            )
            raise ValueError(colored(f"[Config Error] Missing required API keys:\n{err_msg}", "red"))

        
        if not (self.apis.TAVILY_API_KEY or os.getenv("TAVILY_API_KEY")):
            print(colored("[Warning] Missing Tavily API key; search features may not work.", "yellow"))
        if not (self.apis.SEMANTIC_API_KEY or os.getenv("SEMANTIC_API_KEY")):
            print(colored("[Warning] Missing Semantic Scholar API key; literature features may not work.", "yellow"))

        #print(colored("[OK] All required provider API keys validated successfully.", "green"))



def get_settings(config_path: str="config.yaml") -> Settings:
    #global _settings_cache
    _settings_cache: Optional[Settings] = None
    if not os.path.exists(config_path):
        config_path = "src/epidemiqs/config.yaml"
        print(colored(f"[Warning] Config file {config_path} not found, using defaults and .env vars.", "red"))
    if _settings_cache is None:
        try:
            _settings_cache = Settings.from_yaml(config_path)
            
            _settings_cache.validate_provider_secrets()
        except (ValidationError, ValueError) as e:
            raise SystemExit(f"[Config Error] {e}") from e
    return _settings_cache

def reload_settings(config_path: str = "config.yaml") -> Settings:
    global _settings_cache
    _settings_cache = None
    #return get_settings(config_path)
#  LLM  factory 
def make_llm_client(role: Literal["scientists", "experts", "mathematician_expert"]):
    cfg = get_settings()
    print(cfg.query)
    role_cfg = getattr(cfg.llm, role)

    if role_cfg.provider == "openai":
        from openai import OpenAI
        key = cfg.apis.OPENAI_API_KEY.get_secret_value() if cfg.apis.OPENAI_API_KEY else os.getenv("OPENAI_API_KEY")
        return OpenAI(api_key=key), role_cfg.model

    if role_cfg.provider == "grok":
        import requests
        key = cfg.apis.XAI_API_KEY.get_secret_value() if cfg.apis.XAI_API_KEY else os.getenv("XAI_API_KEY")
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {key}"})
        return session, role_cfg.model

    raise RuntimeError(f"Unknown provider: {role_cfg.provider}")


# === Test ===
if __name__ == "__main__":
    print(colored("\n=== Testing configuration loading ===\n", "cyan"))
    cfg = get_settings(config_path="config.yaml")
    print(colored(f"\n{cfg}", "yellow"))
    print(colored("\n[APIs Config]", "yellow"))
    print(cfg.apis.model_dump())
    print(cfg.workflow.scientist_modules.plan)
    try:
        print(colored("Settings loaded successfully!", "green"))
        print(colored(f"Experiment name: {cfg.name}", "blue"))
        print(colored(f"Query: {cfg.query}", "blue"))

        print(colored("\n[Workflow Config]", "yellow"))
        print(cfg.workflow.model_dump())

        print(colored("\n[LLM Config]", "yellow"))
        print(cfg.llm.model_dump())

        print(colored("\n[Paths Config]", "yellow"))
        print(cfg.paths.model_dump())

        for role in ["scientists", "experts", "mathematician_expert"]:
            try:
                client, model = make_llm_client(role)
                print(colored(f" {role} client created with model {model}", "green"))
            except Exception as e:
                print(colored(f" {role} client creation failed: {e}", "red"))

    except Exception as e:
        print(colored(f"[Test Error] {e}", "red"))
    