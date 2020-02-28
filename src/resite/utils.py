from datetime import datetime


def custom_log(message):
    """Prints a given message preceded by current time."""
    print(datetime.now().strftime('%H:%M:%S')+' --- '+str(message))
