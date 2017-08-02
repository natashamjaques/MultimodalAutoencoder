def get_secs_mins_hours_from_secs(total_secs):
    """Compute the hours, minutes, and seconds in a number of seconds.

    Args: 
        total_secs: An integer number of seconds.
    Returns: The hours, minutes, and seconds as ints
    """
    hours = total_secs / 60 / 60
    mins = (total_secs % 3600) / 60
    secs = (total_secs % 3600) % 60

    if hours < 1: hours = 0
    if mins < 1: mins = 0
    
    return hours, mins, secs

def get_friendly_label_name(col):
    if col is None:
        return ""
    if type(col) != str:
        return str(col)

    name = ""
    if 'happiness' in col.lower():
        name ='happiness'
    elif 'calmness' in col.lower():
        name = 'calmness'
    elif 'health' in col.lower():
        name = 'health'

    return name	