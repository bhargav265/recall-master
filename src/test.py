from google.oauth2.credentials import Credentials
from datetime import datetime, timedelta
from pycronofy import Client
import json
import os

def get_calendar_events():
    with open('token.json', 'r') as token_file:
        token_details = token_file.read().strip() 
        
    cronofy_client = Client(access_token=token_details)
    
    one_months_ago = datetime.now() - timedelta(days=30)
    seven_days_future = datetime.now() + timedelta(days=7)

    
    events = cronofy_client.read_events(
        from_date=one_months_ago,
        to_date=seven_days_future,
        tzid='Etc/UTC'
    )

    formatted_events = []
    for event in events:
        formatted_event = {
            'summary': event.get('summary', 'No title'),
            'start': event.get('start', {}),
            'end': event.get('end', {}),
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
    

if __name__ == "__main__":
    print("Fetching calendar events...")
    events = get_calendar_events()
    print("Calendar events:")
    print(events)