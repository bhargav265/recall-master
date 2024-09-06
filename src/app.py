import os
from flask import Flask, render_template, make_response, request
from twilio.twiml.messaging_response import Body, Message, Redirect, MessagingResponse
from chromadb.config import Settings
import chromadb
from uuid import uuid4
import re

app = Flask(__name__)

chroma_client = chromadb.HttpClient(host="chroma", port = 8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))

@app.route('/twilio/incoming_message', methods=['POST'])
def twilio_incoming_message():
    print(request.form)
    from_number = request.form.get('From')
    type_of_comm, number = from_number.split(":")
    incoming_message = request.form.get('Body')
    save_to_chroma(incoming_message, number)
    response = MessagingResponse()
    message = Message()
    message.body('Noted!')
    response.append(message)
    return str(response)


def save_to_chroma(incoming_message, number):
    collection = chroma_client.get_or_create_collection(re.sub('[^A-Za-z0-9]+', '', number))
    collection.add(
    documents=[
        incoming_message
    ],
    ids=[str(uuid4())])
    


@app.errorhandler(404)
def not_found(e):
    return {'err': 'Not found!'}, 404


if __name__ == '__main__':
    port = os.environ.get('PORT', 5001)
    app.run(debug=True, host='0.0.0.0', port=port)
