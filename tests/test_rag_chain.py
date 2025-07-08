# tests/test_rag_chain.py

from unittest.mock import MagicMock, patch
from app.rag_chain import get_rag_chain
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
from langchain_community.chat_models import ChatOpenAI



def test_get_rag_chain_returns_chain(monkeypatch):
    # -- Dummy Retriever that satisfies LangChain's BaseRetriever interface --
    class DummyRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str):
            return []

        async def _aget_relevant_documents(self, query: str):
            return []

    # -- Mock vectorstore with dummy retriever --
    fake_vs = MagicMock()
    fake_vs.as_retriever.return_value = DummyRetriever()
    monkeypatch.setattr("app.rag_chain.get_vectorstore", lambda: fake_vs)

    # -- Dummy LLM that satisfies Runnable --
    class DummyLLM(Runnable):
        def invoke(self, input):
            return "Dummy Answer"

    with patch("app.rag_chain.ChatOpenAI", return_value=DummyLLM()):
        chain = get_rag_chain("openai")
        assert chain is not None
