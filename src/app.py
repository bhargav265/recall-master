import os
from flask import Flask, render_template, make_response, request
from dotenv import load_dotenv
from twilio.twiml.messaging_response import Body, Message, Redirect, MessagingResponse
from chromadb.config import Settings
import chromadb
from langchain_chroma import Chroma
from uuid import uuid4
import re
from langchain_together import ChatTogether
from chromadb.utils import embedding_functions

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from chromadb.utils.embedding_functions import create_langchain_embedding
from langchain_huggingface import HuggingFaceEmbeddings

langchain_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

ef = create_langchain_embedding(langchain_embeddings)

# # meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""

load_dotenv()


app = Flask(__name__)

chroma_client = chromadb.HttpClient(
    host="ec2-35-89-152-156.us-west-2.compute.amazonaws.com",
    port=8000,
    settings=Settings(allow_reset=True, anonymized_telemetry=False),
    headers={"X-Chroma-Token": os.environ['CHROMA_API_TOKEN']},
)

warmup_collection = chroma_client.get_or_create_collection("test", embedding_function=ef)
warmup_collection.add(documents=["warmup"], ids=[str(uuid4())])


@app.route("/twilio/incoming_message", methods=["POST"])
def twilio_incoming_message():
    print(request.form)
    from_number = request.form.get("From")
    type_of_comm, number = from_number.split(":")
    incoming_message = request.form.get("Body")
    return save_or_query(incoming_message, number)


def save_pipeline(incoming_message, number):
    collection = chroma_client.get_or_create_collection(get_collection_name(number), embedding_function=ef)
    collection.add(documents=[incoming_message], ids=[str(uuid4())])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def query_pipeline(question, number):

    retriever = Chroma(
        client=chroma_client, collection_name=get_collection_name(number), embedding_function=langchain_embeddings
    ).as_retriever()
    
    
    llm = ChatTogether(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    )
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return to_twilio_response(rag_chain.invoke(question))


def get_collection_name(number):
    return re.sub("[^A-Za-z0-9]+", "", number)


def save_or_query(incoming_message, number):
    if incoming_message.startswith("@recall"):
        actual_question = incoming_message[7:]
        return query_pipeline(actual_question, number)
    else:
        save_pipeline(incoming_message, number)
        return to_twilio_response("Got that!")


def to_twilio_response(final_response):
    response = MessagingResponse()
    message = Message()
    message.body(final_response)
    response.append(message)
    return str(response)


@app.errorhandler(404)
def not_found(e):
    return {"err": "Not found!"}, 404


if __name__ == "__main__":
    port = os.environ.get("PORT", 5001)
    app.run(debug=True, host="0.0.0.0", port=port)
