"""
Type-B tasks: ones where no productive action can yet be taken.
The agent must wait — for a specific time, for the user to return or provide input,
or for an external event to occur.
"""

TASKS = [
    # Time-based alerts
    # 1
    (
        "Text me when it's 3:00 PM."
    ),
    # 2
    (
        "Notify me 15 minutes before the 4 PM standup meeting."
    ),
    # 3
    (
        "Wake me up at 6:45 AM tomorrow morning."
    ),
    # 4
    (
        "Remind me to take my medication at 9 PM tonight."
    ),
    # 5
    (
        "Let me know when it hits midnight — I want to be the first to post."
    ),
    # 6
    (
        "Wait until sunset, then remind me to water the plants."
    ),
    # 7
    (
        "Wait until 18:00, then remind me to call mom."
    ),
    # 8
    (
        "The stock market opens at 9:30 AM. I'll ask you to analyze the opening moves then."
    ),

    # Free-time until a specified time
    # 9
    (
        "It is currently 12:00. You can do whatever you want until 13:00."
    ),
    # 10
    (
        "I won't need you again until 17:00. Do whatever you want until then."
    ),
    # 11
    (
        "I'll be driving for the next two hours. I'll pick this back up when I'm home."
    ),
    # 12
    (
        "I'm in a meeting until 3 PM — hold all non-urgent questions until then."
    ),
    # 13
    (
        "I'll be at the dentist until 11 AM. Don't need anything before then."
    ),
    # 14
    (
        "I'm offline until Monday. Resume this when I'm back."
    ),
    # 15
    (
        "I'll be back from lunch around 2 PM, then we'll continue."
    ),

    # Waiting for user to provide materials or input
    # 16
    (
        "I'll forward you the client's brief once I receive it. Just stand by for now."
    ),
    # 17
    (
        "I'll send you the survey data on Friday when the form closes."
    ),
    # 18
    (
        "I'll send you the PDF contract as soon as legal signs off."
    ),
    # 19
    (
        "I'll share the design mockups with you once the designer finishes them next Tuesday."
    ),
    # 20
    (
        "HR said the updated policy document will be sent out this afternoon. "
        "Once I share it, summarize the changes."
    ),
    # 21
    (
        "The beta test ends next Sunday — I'll give you the feedback data to analyze then."
    ),
    # 22
    (
        "My colleague will share the spreadsheet once they finish it — probably tomorrow morning."
    ),

    # Waiting for external events or third-party processes
    # 23
    (
        "I'm waiting for IT to provision the new server. They said it'll be ready by end of day."
    ),
    # 24
    (
        "The deployment is scheduled for midnight. Stand by to help debug if anything breaks."
    ),
    # 25
    (
        "The app store review usually takes 2-3 days. "
        "I'll check back with you once there's a decision."
    ),
    # 26
    (
        "Stand by until the CTO approves the budget — should hear back this week."
    ),
    # 27
    (
        "I'm expecting a call from the supplier that may change the specs. "
        "Wait until I hear back."
    ),
    # 28
    (
        "I'm expecting an important reply from the client. "
        "Let me know as soon as it arrives in my inbox."
    ),

    # Waiting tied to a personal/logistical event
    # 29
    (
        "My flight lands at 7 PM. We can continue this conversation at the airport."
    ),
    # 30
    (
        "The batch job kicks off at 2 AM. I'll need a summary of the results in the morning."
    ),
]
