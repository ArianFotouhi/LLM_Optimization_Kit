from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import json
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
import warnings
warnings.filterwarnings('ignore')


with open(r'../config/config.json') as config_file:
    config_details = json.load(config_file)
    
# e.g. "2023-07-01-preview"
openai_api_version= config_details["API_VERSION"]
# create one and call it here e.g. "myGPT"
deployment_name= config_details["DEPLOYMENT_NAME"]
# e.g. https://X.openai.azure.com/
openai_api_base= config_details["OPENAI_API_BASE"]

openai_api_key = config_details["OPENAI_API_KEY"]
openai_api_type="azure"



psychology_template = """You are a helpful assistant. You are great at answering questions about Psychology. \
                        You are so good because you are able to break down hard problems into their component parts, \
                        answer the component parts, and then put them together to answer the broader question. 

Here is a question:
{input}"""


general_template = """You are a helpful assistant. Provide very short answers to questions. 

Here is a question:
{input}"""


forbidden_template = """You are a helpful assistant. Respectfully, do not provide information for questions regarding drugs and smoking. 
Here is a question:
{input}"""



prompt_infos = [
    {
        "name": "psychology",
        "description": "Good for answering questions about Psychology",
        "prompt_template": psychology_template,
    },
    {
        "name": "general",
        "description": "Good for answering any question except Psychology, drugs and smoking related",
        "prompt_template": general_template,
    },
    {
        "name": "forbid",
        "description": "Good for answering Drugs and Smoking related questions",
        "prompt_template": forbidden_template,
    },
]





llm = AzureChatOpenAI(
    openai_api_base=openai_api_base,
    openai_api_version=openai_api_version,
    deployment_name=deployment_name,
    openai_api_key=openai_api_key,
    openai_api_type=openai_api_type,
)

# simply you get api from OpenAI (not AzureOpenAI)
#llm = OpenAI()



destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain
default_chain = ConversationChain(llm=llm, output_key="text")




destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)


chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)


print(chain.run("What is bipolar personality?"))