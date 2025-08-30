from typing import List, Optional, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool, tool
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor

from callbacks import AgentCallbackHandler


@tool
def get_text_length(text: str) -> int:
    """Retorna o comprimento do texto em caracteres."""
    return len(text.strip("'\n").strip('"'))


class SimpleToolCallingAgent:
    """Agente simples usando tool calling nativo via LangChain Agents.

    Exemplo de uso:
        agent = SimpleToolCallingAgent([get_text_length])
        out = agent.run("Qual o tamanho da palavra: DOG")
        print(out)
    """

    def __init__(
        self,
        tools: List[Tool],
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        callbacks: Optional[List[Any]] = None,
        verbose: bool = True,
    ) -> None:
        load_dotenv()

        if system_prompt is None:
            system_prompt = (
                "Você é um agente útil. Use ferramentas quando necessário e responda em português."
            )

        self.tools = tools
        self.callbacks = callbacks or [AgentCallbackHandler()]

        # LLM
        llm = ChatGroq(model=model, temperature=temperature, callbacks=self.callbacks)

        # Prompt com scratchpad para tool calling
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Construção do agente e executor
        agent = create_tool_calling_agent(llm, tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            callbacks=self.callbacks,
            verbose=verbose,
        )

    def run(self, query: str) -> str:
        """Executa o agente com a entrada fornecida e retorna a saída textual."""
        result = self.executor.invoke({"input": query})
        return result.get("output", str(result))


if __name__ == "__main__":
    # Exemplo rápido de execução
    agent = SimpleToolCallingAgent([get_text_length])
    answer = agent.run("Qual é o tamanho da palavra: DOG?")
    print("Resposta:", answer)

