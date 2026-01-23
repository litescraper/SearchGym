from agent.search_r1 import SearchR1Agent  
from agent.SearchGym_reasoning import SearchGymReasoningAgent
from agent.SearchGym_agent import SearchGymAgent


AGENT_CATEGORY = {
    "search-r1": SearchR1Agent,
    "SearchGym-reasoning": SearchGymReasoningAgent,
    "SearchGym": SearchGymAgent,
}

def make_agent(agent_type):
    return AGENT_CATEGORY[agent_type]()