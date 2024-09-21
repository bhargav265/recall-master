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
from pycronofy import Client
from datetime import datetime, timedelta
import json
from langchain_core.prompts import ChatPromptTemplate
langchain_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
from pytz import timezone
from phoenix.otel import register
import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor
import pytz


ef = create_langchain_embedding(langchain_embeddings)


os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "api_key=7d8432aedf54c29f85b:a765f2b"
os.environ["PHOENIX_CLIENT_HEADERS"] = "api_key=7d8432aedf54c29f85b:a765f2b"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"
register(
  project_name="recallmaster", # Default is 'default'
  endpoint="https://app.phoenix.arize.com/v1/traces",
) 

LangChainInstrumentor().instrument()

# # meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo


template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer. Please do not assume as well. 
Focus ONLY on answering the specific question asked. Do not provide any additional information or context that is not directly related to the question.
Use three sentences maximum and keep the answer as concise as possible.
Be confident in your answers based on the calendar events.
If you are to provide a time, then provide the time zone as PST as well.
Do not mention the current date or time unless it's specifically relevant to the question.
Always say "thanks for asking!" at the end of the answer.

{context}

Calendar Events:
{calendar_events}

Question: {question}

Current Time: {current_time}

Helpful Answer:"""

load_dotenv()

app = Flask(__name__)


chroma_client = chromadb.HttpClient(
   host="ec2-54-187-244-103.us-west-2.compute.amazonaws.com",
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
   index = Chroma(
       client=chroma_client, collection_name=get_collection_name(number), embedding_function=langchain_embeddings
   )
   
   
   retriever = index.as_retriever()
  
   calendar_events = get_calendar_events()
  
   current_time = datetime.now().astimezone(timezone('US/Pacific'))
  
   llm = ChatTogether(
       model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
   )
   custom_rag_prompt = PromptTemplate.from_template(template)


   rag_chain = (
       {"context": retriever | format_docs, "question": RunnablePassthrough(), "calendar_events": lambda _: calendar_events, "current_time": lambda _: current_time}
       | custom_rag_prompt
       | llm
       | StrOutputParser()
   )


   return to_twilio_response(rag_chain.invoke(question))

def get_collection_name(number):
   return re.sub("[^A-Za-z0-9]+", "", number)

def save_or_query(incoming_message, number):
   llm = ChatTogether(
       model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
   )
   
   template_for_routing = """
   You are a classifier that determines if a message is a question or information. Respond with only 'question' or 'information'.
   
   For example
   
   1. What is on my calendar?
   Ans: 'question'
   
   2. I have a meeting tomorrow at 2 PM
   Ans: 'information'
   
   3. Looking for a Safe Way to Invest as an NRI inUSA? With ₹18,000/month, you can Get ₹2 Crores!
      Plus, safeguard your family's future with life cover 
      Tap View All Plans and see your wealth grow steadily
      
   Ans: 'information'
   
   {message}
   
   """
  
   classify_prompt = PromptTemplate.from_template(template_for_routing)
  
   classify_chain = classify_prompt | llm | StrOutputParser()
  
   message_type = classify_chain.invoke({"message": incoming_message})
  
   if 'question' in message_type.strip().lower():
       return query_pipeline(incoming_message, number)
   else:
       save_pipeline(incoming_message, number)
       return to_twilio_response("Got that!")

def to_twilio_response(final_response):
   response = MessagingResponse()
   message = Message()
   message.body(final_response)
   response.append(message)
   return str(response)


def get_calendar_events():
      
   cronofy_client = Client(access_token=os.environ['CRONOFY_TOKEN'])
  
   one_months_ago = datetime.now() - timedelta(days=30)
   seven_days_future = datetime.now() + timedelta(days=7)


  
   events = cronofy_client.read_events(
       from_date=one_months_ago,
       to_date=seven_days_future
   )
   pst_timezone = pytz.timezone('America/Los_Angeles')

   formatted_events = []
   for event in events:
       formatted_event = {
           'summary': event.get('summary', 'No title'),
           'start': datetime.strptime(event.get('start'), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc).astimezone(pst_timezone).strftime("%Y-%m-%d %H:%M:%S %Z") if event.get('start') else None,
           'end': datetime.strptime(event.get('end'), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc).astimezone(pst_timezone).strftime("%Y-%m-%d %H:%M:%S %Z") if event.get('end') else None,
           'description': event.get('description', 'No description'),
           'location': event.get('location', {}).get('description', 'No location'),
           'attendees': [attendee.get('email') for attendee in event.get('attendees', [])],
           'organizer': event.get('organizer', {}).get('email', 'No organizer'),
           'id': event.get('event_uid', 'No ID'),
           'calendar_id': event.get('calendar_id', 'No calendar ID'),
           'status': event.get('status', 'No status'),
           'participation_status': event.get('participation_status', 'No participation status'),
           'event_status': event.get('event_status', 'No event status'),
       }
       formatted_events.append(formatted_event)
  
   return json.dumps(formatted_events, indent=2)
  
@app.errorhandler(404)
def not_found(e):
   return {"err": "Not found!"}, 404


if __name__ == "__main__":
   port = os.environ.get("PORT", 5001)
   app.run(debug=True, host="0.0.0.0", port=port)