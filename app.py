import panel as pn
from crewai import Crew, Process, Agent, Task
from langchain_ollama import ChatOllama
from langchain_core.callbacks import BaseCallbackHandler
from crewai.agents import CrewAgentExecutor
from typing import Any, Dict
import threading
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Panel with material design
pn.extension(design="material")

# Define avatars for agents
avatars = {
    "Writer": "https://cdn-icons-png.flaticon.com/512/320/320336.png",
    "Reviewer": "https://cdn-icons-png.freepik.com/512/9408/9408201.png"
}

# Initialize LLM with TinyLlama
llm = ChatOllama(model="tinyllama")

# Custom callback handler for agents
class MyCustomHandler(BaseCallbackHandler):
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        chat_interface.send(inputs['input'], user="assistant", respond=False)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        chat_interface.send(outputs['output'], user=self.agent_name, 
                          avatar=avatars[self.agent_name], respond=False)

# Create agents
writer = Agent(
    role='Blog Post Writer',
    backstory='''You are a blog post writer who is capable of writing a travel blog.
                 You generate one iteration of an article once at a time.
                 You never provide review comments.
                 You are open to reviewer's comments and willing to iterate its article based on these comments.''',
    goal="Write and iterate a decent blog post.",
    llm=llm,
    callbacks=[MyCustomHandler("Writer")],
)

reviewer = Agent(
    role='Blog Post Reviewer',
    backstory='''You are a professional article reviewer and very helpful for improving articles.
                 You review articles and give change recommendations to make the article more aligned with user requests.
                 You will give review comments upon reading entire article, so you will not generate anything when the article is not completely delivered. 
                 You never generate blogs by itself.''',
    goal="list builtins about what need to be improved of a specific blog post. Do not give comments on a summary or abstract of an article",
    llm=llm,
    callbacks=[MyCustomHandler("Reviewer")],
)

# Global variables for handling user input
user_input = None
initiate_chat_task_created = False

def StartCrew(prompt):
    task1 = Task(
        description=f"""Write a blog post of {prompt}. """,
        agent=writer,
        expected_output="an article under 100 words."
    )

    task2 = Task(
        description=("list review comments for improvement from the entire content of blog post to make it more viral on social media."
                    "Make sure to check with a human if your comment is good before finalizing your answer."),
        agent=reviewer,
        expected_output="Builtin points about where need to be improved.",
        human_input=True,
    )

    project_crew = Crew(
        tasks=[task1, task2],
        agents=[writer, reviewer],
        manager_llm=llm,
        process=Process.hierarchical
    )

    result = project_crew.kickoff()
    chat_interface.send("## Final Result\n"+result, user="assistant", respond=False)

def initiate_chat(message):
    global initiate_chat_task_created
    initiate_chat_task_created = True
    StartCrew(message)

def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    global initiate_chat_task_created
    global user_input

    if not initiate_chat_task_created:
        thread = threading.Thread(target=initiate_chat, args=(contents,))
        thread.start()
    else:
        user_input = contents

# Custom human input function
def custom_ask_human_input(self, final_answer: dict) -> str:
    global user_input

    prompt = self._i18n.slice("getting_input").format(final_answer=final_answer)
    chat_interface.send(prompt, user="assistant", respond=False)

    while user_input == None:
        time.sleep(1)

    human_comments = user_input
    user_input = None
    return human_comments

# Override CrewAgentExecutor's _ask_human_input method
CrewAgentExecutor._ask_human_input = custom_ask_human_input

# Initialize chat interface
chat_interface = pn.chat.ChatInterface(callback=callback)
chat_interface.send("Send a message!", user="System", respond=False)
chat_interface.servable() 