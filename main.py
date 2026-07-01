import os
import requests
from groq import Groq
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CRICKET_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="CricAI Backend")


@app.get("/")
def root():
    return {"message": "CricAI backend is running"}


@app.get("/matches")
def get_matches():
    url = "https://api.cricapi.com/v1/currentMatches"
    params = {"apikey": API_KEY, "offset": 0}
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "success":
        return {"error": "Failed to fetch matches"}

    matches = []
    for m in data.get("data", []):
        matches.append({
            "id": m.get("id"),
            "name": m.get("name"),
            "status": m.get("status"),
            "venue": m.get("venue"),
            "teams": m.get("teams"),
            "score": m.get("score"),
            "matchStarted": m.get("matchStarted"),
            "matchEnded": m.get("matchEnded"),
        })

    return {"count": len(matches), "matches": matches}


def generate_ai_insight(teams, status, score):
    prompt = f"""You are a sharp cricket analyst writing for a mobile app called CricAI.
Given this match data, write a short, insightful 2-3 sentence summary that explains
WHY the result happened — not just what the score was. Be specific, use the numbers,
and sound like a knowledgeable commentator, not a stat sheet.

Teams: {teams}
Status: {status}
Score breakdown: {score}

Write only the insight, no preamble."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI insight unavailable: {str(e)}"


@app.get("/insights/{match_id}")
def get_insights(match_id: str):
    url = "https://api.cricapi.com/v1/match_info"
    params = {"apikey": API_KEY, "id": match_id}
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "success":
        return {"error": "Failed to fetch match info"}

    match = data.get("data", {})
    teams = match.get("teams", [])
    score = match.get("score", [])
    status = match.get("status", "")

    ai_summary = generate_ai_insight(teams, status, score)

    return {
        "match_id": match_id,
        "teams": teams,
        "status": status,
        "ai_insight": ai_summary,
        "score_breakdown": score,
    }


def generate_prematch_insight(teams, venue, toss_winner, toss_choice):
    prompt = f"""You are a sharp cricket analyst writing a pre-match preview for the CricAI app.
Given the details below, write a short, engaging 2-3 sentence "what to watch for" prediction.
Mention any tactical edge implied by the toss decision or venue, and what fans should expect.
If information is limited, focus on what IS known rather than inventing details.

Teams: {teams}
Venue: {venue}
Toss winner: {toss_winner}
Toss decision: {toss_choice}

Write only the insight, no preamble."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI insight unavailable: {str(e)}"


@app.get("/pre-match/{match_id}")
def get_prematch(match_id: str):
    url = "https://api.cricapi.com/v1/match_info"
    params = {"apikey": API_KEY, "id": match_id}
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "success":
        return {"error": "Failed to fetch match info"}

    match = data.get("data", {})
    teams = match.get("teams", [])
    venue = match.get("venue", "Unknown venue")
    toss_winner = match.get("tossWinner", "Not available yet")
    toss_choice = match.get("tossChoice", "Not available yet")

    ai_preview = generate_prematch_insight(teams, venue, toss_winner, toss_choice)

    return {
        "match_id": match_id,
        "teams": teams,
        "venue": venue,
        "toss_winner": toss_winner,
        "toss_choice": toss_choice,
        "ai_preview": ai_preview,
    }


def generate_match_report(teams, status, score, venue):
    prompt = f"""You are CricAI's lead match analyst, writing a complete post-match report
for the CricAI app — the kind of in-depth report a professional cricket journalist would write,
not a simple scorecard recap.

Match data:
Teams: {teams}
Venue: {venue}
Result: {status}
Score breakdown: {score}

Write a structured match report with these exact sections, using the headers below:

SUMMARY:
A 2-3 sentence overview of how the match unfolded and the final result.

TURNING POINT:
Identify the single most decisive phase or moment of the match (based on the score
breakdown — e.g. a low-scoring innings, a collapse, a big over count vs wickets ratio)
and explain why it decided the outcome. Be specific using the numbers given.

KEY PERFORMERS:
Based on the score data available, note which team/innings stood out and why
(e.g. strong run rate, depth in batting). If individual player data isn't available,
focus on team-level standout performances.

TACTICAL READ:
A short tactical explanation of why the winning side prevailed — pacing, run rate
pressure, required rate, or similar reasoning grounded in the actual numbers.

Keep the tone sharp, confident, and analytical — like a knowledgeable cricket
journalist, not generic filler text. Do not invent specific player names or stats
that aren't in the data provided."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI match report unavailable: {str(e)}"


@app.get("/match-report/{match_id}")
def get_match_report(match_id: str):
    url = "https://api.cricapi.com/v1/match_info"
    params = {"apikey": API_KEY, "id": match_id}
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "success":
        return {"error": "Failed to fetch match info"}

    match = data.get("data", {})
    teams = match.get("teams", [])
    score = match.get("score", [])
    status = match.get("status", "")
    venue = match.get("venue", "Unknown venue")
    match_ended = match.get("matchEnded", False)

    if not match_ended:
        return {
            "match_id": match_id,
            "teams": teams,
            "status": status,
            "message": "Match report will be available once the match concludes.",
        }

    report = generate_match_report(teams, status, score, venue)

    return {
        "match_id": match_id,
        "teams": teams,
        "venue": venue,
        "status": status,
        "match_report": report,
    }


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


def find_relevant_match(message, matches):
    message_lower = message.lower()
    for m in matches:
        for team in m.get("teams", []):
            if team.lower() in message_lower:
                return m
    return None


@app.post("/chat")
def chat(request: ChatRequest):
    url = "https://api.cricapi.com/v1/currentMatches"
    params = {"apikey": API_KEY, "offset": 0}
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "success":
        return {"error": "Failed to fetch matches"}

    matches = data.get("data", [])

    relevant_match = find_relevant_match(request.message, matches)

    if relevant_match:
        context = f"""Relevant match data:
Teams: {relevant_match.get('teams')}
Status: {relevant_match.get('status')}
Score: {relevant_match.get('score')}
Venue: {relevant_match.get('venue')}"""
    else:
        summary_list = [f"{m.get('name')}: {m.get('status')}" for m in matches[:10]]
        context = "Recent matches:\n" + "\n".join(summary_list)

    system_prompt = f"""You are CricAI, a knowledgeable cricket assistant inside a mobile app.
Answer questions using the match data below when relevant. Be concise, specific, and
conversational — like a sharp cricket friend, not a stats dump. If the data doesn't
contain what's needed to answer, say so honestly. Use the conversation history to
understand follow-up questions and context.

{context}"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in request.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.message})

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=250,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Sorry, I couldn't process that: {str(e)}"

    return {"question": request.message, "answer": answer}