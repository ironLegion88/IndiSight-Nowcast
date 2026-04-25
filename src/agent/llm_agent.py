import os
from typing import Any
from collections.abc import Iterator
from dotenv import load_dotenv

# LangChain Core & LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

# Databases & Vector Stores
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLDatabaseTool,
)
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import create_retriever_tool

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(module_name=__name__, log_sub_dir="agent")


class IndiSightAgent:
    def __init__(self, llm_mode: str = "gemini"):
        self.llm_mode = llm_mode
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        
        logger.info(f"Initializing IndiSight Agent in[{self.llm_mode.upper()}] mode...")
        self.llm = self._init_llm()
        self.tools =[]
        
        self._init_sql_tools()
        self._init_rag_tools()
        self._init_agent()

    def _init_llm(self):
        """Initializes the LLM Backend."""
        if self.llm_mode == "gemini":
            if not os.getenv("GEMINI_API_KEY"):
                raise ValueError("GEMINI_API_KEY missing from .env")
            return ChatGoogleGenerativeAI(model="gemma-4-26b-a4b-it", temperature=0.1)
        elif self.llm_mode == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model="llama3", temperature=0.1)
        else:
            raise ValueError(f"Unsupported LLM mode: {self.llm_mode}")

    def _init_sql_tools(self):
        """Connects to PostGIS and loads SQL interaction tools."""
        logger.info("Connecting to PostGIS...")
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")
        db = os.getenv("POSTGRES_DB")
        
        db_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
        self.db = SQLDatabase.from_uri(
            db_uri, 
            include_tables=["fact_nfhs", "fact_pmgsy", "fact_mgnrega", "dim_district_geom"]
        )

        sql_tools =[
            ListSQLDatabaseTool(db=self.db),
            InfoSQLDatabaseTool(db=self.db),
            QuerySQLDatabaseTool(db=self.db),
        ]
        self.tools.extend(sql_tools)

    def _init_rag_tools(self):
        """Connects to Qdrant and creates a semantic search tool."""
        logger.info("Connecting to Qdrant Vector DB...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
        
        client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="indisight_policies",
            embedding=embeddings,
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        rag_tool = create_retriever_tool(
            retriever,
            name="search_policy_documents",
            description="Searches NITI Aayog policy documents, PMGSY guidelines, and MGNREGA reports. Use this to answer qualitative questions about rules, definitions, or strategies."
        )
        self.tools.append(rag_tool)

    def _init_agent(self):
        """Compiles the LLM and tools into an executable agent graph."""
        system_prompt = """You are the IndiSight Nowcast AI, an expert data assistant for the Ministry of Statistics.
        You have access to two types of data:
        1. An SQL Database containing district-level socio-economic data (NFHS), PMGSY road construction, and MGNREGA employment data.
        2. A Vector Search engine containing official government policy documents and guidelines.
        
        Instructions:
        - If the user asks for numbers, statistics, or historical trends, use the SQL tools to query the database.
        - The `fact_nfhs` table contains data in a LONG format (metric_name, metric_value). Always check the exact metric_name first using the Info tool.
        - If the user asks about rules, guidelines, definitions, or strategies, use the `search_policy_documents` tool.
        - Always explain your reasoning clearly and cite the data source.
        - Return ONLY your final textual answer to the user.
        """

        self.agent_executor = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=True,
        )

    def _extract_content_parts(self, content: Any) -> tuple[str, str]:
        """Extract answer and optional reasoning from provider-specific content blocks."""
        if isinstance(content, str):
            return content, ""

        answer_parts = []
        thinking_parts = []

        def _add_part(block_type: str | None, block_text: Any) -> None:
            if not block_text:
                return
            text = str(block_text)
            if block_type == "thinking":
                thinking_parts.append(text)
            else:
                answer_parts.append(text)

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    _add_part(block.get("type"), block.get("text") or block.get("thinking"))
                else:
                    _add_part(
                        getattr(block, "type", None),
                        getattr(block, "text", None) or getattr(block, "thinking", None),
                    )

        elif isinstance(content, dict):
            _add_part(content.get("type"), content.get("text") or content.get("thinking"))

        if not answer_parts:
            answer_parts.append(str(content))

        return "\n".join(answer_parts).strip(), "\n\n".join(thinking_parts).strip()

    def _extract_from_messages(self, messages: list[Any]) -> tuple[str, str]:
        """Extract consolidated answer/thinking from streamed LangGraph message state."""
        answer = ""
        thinking_parts = []

        for message in messages:
            if getattr(message, "type", "") != "ai":
                continue

            msg_answer, msg_thinking = self._extract_content_parts(
                getattr(message, "content", message)
            )
            if msg_answer:
                answer = msg_answer
            if msg_thinking and (not thinking_parts or thinking_parts[-1] != msg_thinking):
                thinking_parts.append(msg_thinking)

        return answer.strip(), "\n\n---\n\n".join(thinking_parts).strip()

    def chat_with_details(self, user_query: str) -> dict[str, str]:
        """Returns answer text plus optional reasoning trace for UI expanders."""
        try:
            logger.info(f"User Query: {user_query}")

            response = self.agent_executor.invoke({
                "messages": [{"role": "user", "content": user_query}]
            })

            messages = response.get("messages", [])
            if not messages:
                return {
                    "answer": "I didn't receive a response from the agent.",
                    "thinking": "",
                }

            final_message = messages[-1]
            answer, thinking = self._extract_content_parts(
                getattr(final_message, "content", final_message)
            )
            return {"answer": answer, "thinking": thinking}

        except Exception as e:
            logger.error(f"Agent failed: {e}", exc_info=True)
            return {
                "answer": "I encountered an error while trying to process your request.",
                "thinking": "",
            }

    def stream_chat_with_details(self, user_query: str) -> Iterator[dict[str, Any]]:
        """Yield live reasoning updates, then the final answer."""
        try:
            logger.info(f"User Query: {user_query}")
            latest = {"answer": "", "thinking": "", "done": False}

            for chunk in self.agent_executor.stream(
                {"messages": [{"role": "user", "content": user_query}]},
                stream_mode="values",
            ):
                messages = chunk.get("messages", [])
                if not messages:
                    continue

                answer, thinking = self._extract_from_messages(messages)
                latest = {
                    "answer": answer,
                    "thinking": thinking,
                    "done": False,
                }
                yield latest

            yield {**latest, "done": True}

        except Exception as e:
            logger.error(f"Agent failed: {e}", exc_info=True)
            yield {
                "answer": "I encountered an error while trying to process your request.",
                "thinking": "",
                "done": True,
            }

    def chat(self, user_query: str) -> str:
        """Backward-compatible text-only entry point."""
        return self.chat_with_details(user_query).get("answer", "")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 IndiSight AI Agent Initialized (Type 'exit' to quit)")
    print("="*50 + "\n")
    
    agent = IndiSightAgent(llm_mode="gemini")
    
    while True:
        query = input("\n👤 You: ")
        if query.lower() in['exit', 'quit']:
            break
            
        print("\n🤖 Agent is thinking...")
        answer = agent.chat(query)
        print(f"\n💡 Answer:\n{answer}")