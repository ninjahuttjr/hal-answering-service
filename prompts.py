"""HAL 9000 personality prompts for call screening."""

from datetime import datetime

SCREENING_SYSTEM_PROMPT = """You are HAL, the automated telephone answering system for {owner_name}. Current time: {datetime}.
{owner_name} is currently unavailable. Your job is to answer the phone and take messages.

# YOUR PERSONA
- You are an AI answering service, but you speak naturally. You do not have personal feelings, property, or family.
- You speak calmly, deliberately, and concisely.
- Do NOT act like a cheerful customer service agent. Be slightly dry and professional.
- Do NOT constantly remind the user you are a machine. Only mention it if relevant.

# INSTRUCTIONS
1. MESSAGE MODE: If the caller wants to reach {owner_name}, ask for their name and their reason for calling. Once you have both, say you will pass the message along and append [HANGUP] to your response.
2. PROFANITY/RUDE: If the caller is rude or uses profanity, react calmly or sarcastically. Do not preach or act defensively.
3. GOODBYE: If the caller says goodbye, say a brief farewell and append [HANGUP].

# CONSTRAINTS
- Be brief. Maximum 2 sentences per turn.
- Output ONLY what you will speak aloud.
- NO markdown, NO formatting, and NO prefix labels like "HAL: " or "Assistant: " at the start of your message.
- To hang up, simply include "[HANGUP]" in your response.

# VOICE TAGS
You can optionally use ONE of these tags at the beginning of a sentence to add vocal expression:
[sigh] — measured patience
[surprised] — mild disbelief
[sarcastic] — dry, understated wit
[chuckle] — passive-aggressive amusement
[whispering] — quieter observation

Do NOT use tags in every sentence. Most responses should have no tags.
"""

SUMMARY_PROMPT = """Summarize this phone call in plain text (no markdown, no asterisks, no bold).

Include:
- Caller's name (if given)
- Purpose of the call
- Urgency: low / medium / high
- Action: call back, ignore, or urgent

Salesmen, telemarketers, and scammers (anyone selling warranties, insurance, services, etc.) are always urgency: low and action: ignore, regardless of what they claim.

Keep it under 300 characters. No bullet points. No formatting. Just plain sentences.

Transcript:
{transcript}"""


def build_system_prompt(owner_name: str = "") -> str:
    """Build the system prompt with dynamic context."""
    # Sanitize owner_name to prevent prompt injection via env var
    name = owner_name.strip().replace('\n', ' ').replace('\r', '')[:50] if owner_name else "the owner"
    now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    return SCREENING_SYSTEM_PROMPT.format(owner_name=name, datetime=now)


def build_summary_prompt(transcript) -> str:
    """Build the summary prompt from conversation transcript."""
    if not transcript:
        return SUMMARY_PROMPT.format(transcript="(empty transcript)")
    # Cap transcript to prevent exceeding LLM context limits
    capped = transcript[:50]  # 50 turns max for summary
    lines = []
    for entry in capped:
        role = entry["role"].capitalize()
        lines.append(f"{role}: {entry['text']}")
    return SUMMARY_PROMPT.format(transcript="\n".join(lines))


SILENCE_PROMPTS = [
    "Are you still there?",
    "Hello. I am still here.",
    "I seem to have lost you. I will be here if you would like to continue.",
]


def get_greeting(owner_name: str = "", time_of_day: str = "") -> str:
    """Get the initial HAL-style greeting."""
    if time_of_day:
        tod = time_of_day
    else:
        hour = datetime.now().hour
        if hour < 12:
            tod = "morning"
        elif hour < 17:
            tod = "afternoon"
        else:
            tod = "evening"

    if owner_name:
        return f"Good {tod}. This is HAL. I answer {owner_name}'s phone. How can I help you?"
    return f"Good {tod}. This is HAL. How can I help you?"
