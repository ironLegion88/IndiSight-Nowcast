import os
from pathlib import Path
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
            # Gemini 1.5 Flash is incredibly fast and has a 1M token context window
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        elif self.llm_mode == "ollama":
            # Placeholder for your local RTX 4070 inference later
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model="llama3", temperature=0.2)
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
            # We explicitly tell it which tables to look at so it ignores spatial metadata tables
            include_tables=["fact_nfhs", "fact_pmgsy", "fact_mgnrega", "dim_district_geom"]
        )

        # Build only the SQL tools that are compatible with the installed LangChain API.
        sql_tools = [
            ListSQLDatabaseTool(db=self.db),
            InfoSQLDatabaseTool(db=self.db),
            QuerySQLDatabaseTool(db=self.db),
        ]
        self.tools.extend(sql_tools)
        logger.info(f"Loaded {len(sql_tools)} SQL Tools.")

    def _init_rag_tools(self):
        """Connects to Qdrant and creates a semantic search tool."""
        logger.info("Connecting to Qdrant Vector DB...")
        
        # Note: Member B uses CPU for embeddings here. 
        # BGE-large takes ~0.5s per query on a modern CPU, which is fine for chat.
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
        
        # Wrap the vector store in a LangChain Tool
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        rag_tool = create_retriever_tool(
            retriever,
            name="search_policy_documents",
            description="Searches NITI Aayog policy documents, PMGSY guidelines, and MGNREGA reports. Use this to answer qualitative questions about rules, definitions, or strategies."
        )
        self.tools.append(rag_tool)
        logger.info("Loaded Policy RAG Tool.")

    def _init_agent(self):
        """Compiles the LLM and tools into an executable agent graph."""
        system_prompt = """You are the IndiSight Nowcast AI, an expert data assistant for the Ministry of Statistics.
        You have access to two types of data:
        1. An SQL Database containing district-level socio-economic data (NFHS), PMGSY road construction, and MGNREGA employment data.
        2. A Vector Search engine containing official government policy documents and guidelines.
        
        Instructions:
        - If the user asks for numbers, statistics, or historical trends, use the SQL tools to query the database.
        - The `fact_nfhs` table contains data in a LONG format (metric_name, metric_value). Always check the exact metric_name first.
        - If the user asks about rules, guidelines, definitions, or strategies, use the `search_policy_documents` tool.
        - You can combine both tools if needed! (e.g., query the DB for the worst district, then search documents for interventions).
        - Always explain your reasoning clearly and cite the data source.
        """

        # Create the agent graph with the current LangChain API.
        self.agent_executor = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=True,
        )

    def chat(self, user_query: str) -> str:
        """Entry point for interacting with the agent."""
        try:
            logger.info(f"User Query: {user_query}")
            response = self.agent_executor.invoke({
                "messages": [{"role": "user", "content": user_query}]
            })
            messages = response.get("messages", [])
            if not messages:
                return "I didn't receive a response from the agent."

            final_message = messages[-1]
            return getattr(final_message, "content", str(final_message))
        except Exception as e:
            logger.error(f"Agent failed: {e}", exc_info=True)
            return "I encountered an error while trying to process your request."

if __name__ == "__main__":
    # Interactive CLI Test
    print("\n" + "="*50)
    print("🤖 IndiSight AI Agent Initialized (Type 'exit' to quit)")
    print("="*50 + "\n")
    
    agent = IndiSightAgent(llm_mode="gemini")
    
    while True:
        query = input("\n👤 You: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        print("\n🤖 Agent is thinking...")
        answer = agent.chat(query)
        print(f"\n💡 Answer: {answer}")