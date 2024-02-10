
def analyse_browser(user_agent):
    if "Android" in user_agent:
        return "Android"
    elif "iPhone" in user_agent:
        return "iPhone"
    elif "iPad" in user_agent:
        return "iPad"
    elif "Opera" in user_agent:
        return  "Opera"
    elif "Firefox" in user_agent:
        return "Firefox"
    elif "Chrome" in user_agent:
        return "Chrome"
    elif "Safari" in user_agent:
        return "Safari"
    return user_agent
