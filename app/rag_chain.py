from langchain.chains import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub, LlamaCpp
from app.ingestion import get_vectorstore


def get_rag_chain(model_name="OpenAI"):
    """
    Returns a RetrievalQA chain with a custom prompt and selected LLM backend.

    Args:
        model_name (str): One of ["OpenAI", "HuggingFace", "LLaMA"].

    Raises:
        ValueError: If vectorstore is empty or unsupported model_name.

    Returns:
        RetrievalQAWithSourcesChain: A LangChain QA chain with document retrieval.
    """
    vectorstore = get_vectorstore()
    if vectorstore is None:
        raise ValueError("Vectorstore is empty. Upload documents first.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Set up prompt with assistant role and context placeholder
    system_message = SystemMessagePromptTemplate.from_template(
        "You are a proactive and friendly AI assistant helping users understand their documents. "
        "You answer questions based on the provided context, but you also think ahead: "
        "- If information is missing, point it out clearly and suggest what the user could do. "
        "- If the question is unclear, ask clarifying questions. "
        "- If the answer requires explanation, provide a detailed but understandable response. "
        "- Suggest relevant next steps, improvements, or warnings if needed. "
        "Do not make up information. If the context doesn't contain the answer, say you don't know."
    )


    human_message = HumanMessagePromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion:\n{question}"
    )

    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    # Select the LLM
    model_name = model_name.lower()

    if model_name == "openai":
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    elif model_name == "huggingface":
        llm = HuggingFaceHub(
            repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.5}
        )
    elif model_name == "llama":
        llm = LlamaCpp(
            model_path="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
            n_ctx=2048,
            temperature=0.7,
            verbose=False,
        )
    else:
        raise ValueError(f"Unsupported model selected: {model_name}")

    # Build the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

    return qa_chain
