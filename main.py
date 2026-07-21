import os
import json
import time
import requests
from groq import Groq
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CRICKET_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="CricAI Backend")

# ─── Simple In-Memory Cache ───
_cache = {}

def cache_get(key: str):
    if key in _cache:
        data, expiry = _cache[key]
        if time.time() < expiry:
            return data
        del _cache[key]
    return None

def cache_set(key: str, data, ttl_seconds: int):
    _cache[key] = (data, time.time() + ttl_seconds)

def cached_cricket_request(url: str, params: dict, ttl: int = 60):
    cache_key = url + str(sorted(params.items()))
    cached = cache_get(cache_key)
    if cached:
        return cached
    response = requests.get(url, params=params)
    data = response.json()
    if data.get("status") == "success":
        cache_set(cache_key, data, ttl)
    return data


@app.get("/")
def root():
    return {"message": "CricAI backend is running"}


@app.get("/cache/status")
def cache_status():
    return {
        "cached_keys": len(_cache),
        "keys": list(_cache.keys())
    }


@app.get("/matches")
def get_matches():
    url = "https://api.cricapi.com/v1/currentMatches"
    params = {"apikey": API_KEY, "offset": 0}
    data = cached_cricket_request(url, params, ttl=60)

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
    cache_key = f"insights_{match_id}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    url = "https://api.cricapi.com/v1/match_info"
    params = {"apikey": API_KEY, "id": match_id}
    data = cached_cricket_request(url, params, ttl=120)

    if data.get("status") != "success":
        return {"error": "Failed to fetch match info"}

    match = data.get("data", {})
    teams = match.get("teams", [])
    score = match.get("score", [])
    status = match.get("status", "")

    ai_summary = generate_ai_insight(teams, status, score)

    result = {
        "match_id": match_id,
        "teams": teams,
        "status": status,
        "ai_insight": ai_summary,
        "score_breakdown": score,
    }
    cache_set(cache_key, result, 120)
    return result


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
    cache_key = f"prematch_{match_id}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    url = "https://api.cricapi.com/v1/match_info"
    params = {"apikey": API_KEY, "id": match_id}
    data = cached_cricket_request(url, params, ttl=120)

    if data.get("status") != "success":
        return {"error": "Failed to fetch match info"}

    match = data.get("data", {})
    teams = match.get("teams", [])
    venue = match.get("venue", "Unknown venue")
    toss_winner = match.get("tossWinner", "Not available yet")
    toss_choice = match.get("tossChoice", "Not available yet")

    ai_preview = generate_prematch_insight(teams, venue, toss_winner, toss_choice)

    result = {
        "match_id": match_id,
        "teams": teams,
        "venue": venue,
        "toss_winner": toss_winner,
        "toss_choice": toss_choice,
        "ai_preview": ai_preview,
    }
    cache_set(cache_key, result, 300)
    return result


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
    cache_key = f"report_{match_id}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    url = "https://api.cricapi.com/v1/match_info"
    params = {"apikey": API_KEY, "id": match_id}
    data = cached_cricket_request(url, params, ttl=120)

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

    result = {
        "match_id": match_id,
        "teams": teams,
        "venue": venue,
        "status": status,
        "match_report": report,
    }
    cache_set(cache_key, result, 3600)
    return result


class ChatMessage(BaseModel):
    role: str
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
    data = cached_cricket_request(url, params, ttl=60)

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


@app.get("/news")
def get_news():
    cache_key = "news_all"
    cached = cache_get(cache_key)
    if cached:
        return cached

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "cricket",
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "ok":
        return {"error": "Failed to fetch news"}

    articles = []
    for a in data.get("articles", []):
        if a.get("title") and a.get("url"):
            articles.append({
                "title": a.get("title"),
                "description": a.get("description", ""),
                "url": a.get("url"),
                "source": a.get("source", {}).get("name", ""),
                "publishedAt": a.get("publishedAt", ""),
                "urlToImage": a.get("urlToImage", ""),
            })

    result = {"count": len(articles), "articles": articles}
    cache_set(cache_key, result, 1800)
    return result


def summarize_news(title, description):
    if not description:
        return None

    prompt = f"""You are CricAI's news editor. Summarize this cricket news in 2 sharp sentences.
Be specific, use key facts, and sound like a knowledgeable cricket journalist.

Title: {title}
Description: {description}

Write only the summary, no preamble."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return description


@app.get("/news/summary")
def get_news_summary():
    cache_key = "news_summary"
    cached = cache_get(cache_key)
    if cached:
        return cached

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "cricket",
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "ok":
        return {"error": "Failed to fetch news"}

    articles = []
    for a in data.get("articles", []):
        if not a.get("title") or not a.get("url"):
            continue
        ai_summary = summarize_news(a.get("title"), a.get("description", ""))
        articles.append({
            "title": a.get("title"),
            "ai_summary": ai_summary,
            "url": a.get("url"),
            "source": a.get("source", {}).get("name", ""),
            "publishedAt": a.get("publishedAt", ""),
            "urlToImage": a.get("urlToImage", ""),
        })

    result = {"count": len(articles), "articles": articles}
    cache_set(cache_key, result, 1800)
    return result


@app.get("/series")
def get_series():
    cache_key = "series_all"
    cached = cache_get(cache_key)
    if cached:
        return cached

    url = "https://api.cricapi.com/v1/series"
    params = {"apikey": API_KEY, "offset": 0}
    data = cached_cricket_request(url, params, ttl=3600)

    if data.get("status") != "success":
        return {"error": "Failed to fetch series"}

    series_list = []
    for s in data.get("data", []):
        series_list.append({
            "id": s.get("id"),
            "name": s.get("name"),
            "startDate": s.get("startDate"),
            "endDate": s.get("endDate"),
            "odi": s.get("odi", 0),
            "t20": s.get("t20", 0),
            "test": s.get("test", 0),
            "squads": s.get("squads", 0),
            "matches": s.get("matches", 0),
        })

    result = {"count": len(series_list), "series": series_list}
    cache_set(cache_key, result, 3600)
    return result


@app.get("/series/{series_id}/points")
def get_series_points(series_id: str):
    cache_key = f"points_{series_id}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    url = "https://api.cricapi.com/v1/series_points"
    params = {"apikey": API_KEY, "id": series_id}
    data = cached_cricket_request(url, params, ttl=1800)

    if data.get("status") != "success":
        return {"error": "Points table not available for this series"}

    result = {
        "series_id": series_id,
        "points_table": data.get("data", [])
    }
    cache_set(cache_key, result, 1800)
    return result


@app.get("/series/{series_id}/matches")
def get_series_matches(series_id: str):
    cache_key = f"series_matches_{series_id}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    url = "https://api.cricapi.com/v1/series_info"
    params = {"apikey": API_KEY, "id": series_id}
    data = cached_cricket_request(url, params, ttl=1800)

    if data.get("status") != "success":
        return {"error": "Failed to fetch series info"}

    info = data.get("data", {})
    matches = info.get("matchList", [])

    series_matches = []
    for m in matches:
        series_matches.append({
            "id": m.get("id"),
            "name": m.get("name"),
            "date": m.get("date"),
            "dateTimeGMT": m.get("dateTimeGMT"),
            "teams": m.get("teams"),
            "venue": m.get("venue"),
            "status": m.get("status"),
            "matchType": m.get("matchType"),
        })

    result = {
        "series_id": series_id,
        "name": info.get("info", {}).get("name", ""),
        "matches": series_matches,
    }
    cache_set(cache_key, result, 1800)
    return result


@app.get("/rankings/{format}")
def get_rankings(format: str):
    cache_key = f"rankings_{format}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    prompt = f"""You are a cricket data assistant. Provide the current ICC {format.upper()} rankings.

Return ONLY a JSON array with exactly this structure, no other text:
[
  {{"rank": 1, "team": "Australia", "rating": 128}},
  {{"rank": 2, "team": "India", "rating": 121}},
  ...up to rank 10
]

For batting rankings also include player name and country.
Be as accurate as possible based on your latest knowledge."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        text = response.choices[0].message.content.strip()
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != 0:
            rankings = json.loads(text[start:end])
        else:
            rankings = []
        result = {
            "format": format,
            "rankings": rankings,
            "note": "Rankings based on AI knowledge — updated periodically"
        }
        cache_set(cache_key, result, 86400)
        return result
    except Exception as e:
        return {"error": f"Failed to generate rankings: {str(e)}"}


@app.get("/scorecard/{match_id}")
def get_scorecard(match_id: str):
    cache_key = f"scorecard_{match_id}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    url = "https://api.cricapi.com/v1/match_scorecard"
    params = {"apikey": API_KEY, "id": match_id}
    data = cached_cricket_request(url, params, ttl=120)

    if data.get("status") != "success":
        return {"error": "Detailed scorecard not available"}

    result = {"match_id": match_id, "scorecard": data.get("data", {})}
    cache_set(cache_key, result, 120)
    return result


def generate_motm_prediction(teams, status, score):
    prompt = f"""You are CricAI's match analyst. Based on this match data, predict who was
the most valuable player (Man of the Match).

Teams: {teams}
Result: {status}
Scores: {score}

Respond in exactly this JSON format, no other text:
{{"predicted_motm": "Player Name", "team": "Team Name", "reasoning": "2 sentence explanation"}}

If you cannot determine a specific player from the data, make an educated guess based
on the winning team and match context. Do not say unknown."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )
        text = response.choices[0].message.content.strip()
        start = text.find('{')
        end = text.rfind('}') + 1
        return json.loads(text[start:end])
    except Exception:
        return {
            "predicted_motm": "Top performer",
            "team": teams[0] if teams else "",
            "reasoning": "Based on match context and winning team performance"
        }


@app.get("/motm/{match_id}")
def get_motm(match_id: str):
    cache_key = f"motm_{match_id}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    url = "https://api.cricapi.com/v1/match_info"
    params = {"apikey": API_KEY, "id": match_id}
    data = cached_cricket_request(url, params, ttl=120)

    if data.get("status") != "success":
        return {"error": "Failed to fetch match info"}

    match = data.get("data", {})
    teams = match.get("teams", [])
    score = match.get("score", [])
    status = match.get("status", "")
    official_motm = match.get("playerOfMatch", None)
    official_motm_team = match.get("playerOfMatchTeam", None)

    ai_motm = generate_motm_prediction(teams, status, score)

    result = {
        "match_id": match_id,
        "official_motm": official_motm,
        "official_motm_team": official_motm_team,
        "ai_motm": ai_motm,
    }
    cache_set(cache_key, result, 3600)
    return result


@app.get("/predict/featured")
def get_featured_match():
    cache_key = "featured_match"
    cached = cache_get(cache_key)
    if cached:
        return cached

    url = "https://api.cricapi.com/v1/currentMatches"
    params = {"apikey": API_KEY, "offset": 0}
    data = cached_cricket_request(url, params, ttl=60)

    if data.get("status") != "success":
        return {"error": "Failed to fetch matches"}

    matches = data.get("data", [])
    live = [m for m in matches if m.get("matchStarted") and not m.get("matchEnded")]
    upcoming = [m for m in matches if not m.get("matchStarted") and not m.get("matchEnded")]
    recent = [m for m in matches if m.get("matchStarted") and m.get("matchEnded")]

    featured = live[0] if live else (upcoming[0] if upcoming else (recent[0] if recent else None))

    if not featured:
        return {"error": "No featured match available"}

    teams = featured.get("teams", [])
    score = featured.get("score", [])
    status = featured.get("status", "")

    prompt = f"""You are CricAI's prediction engine. Based on this match data, predict the winner
and give a confidence percentage. Be specific and data-driven.

Teams: {teams}
Status: {status}
Score: {score}

Respond in exactly this JSON format, no other text:
{{"predicted_winner": "Team Name", "confidence": 65, "reasoning": "2 sentence reason"}}"""

    try:
        ai_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )
        text = ai_response.choices[0].message.content.strip()
        start = text.find('{')
        end = text.rfind('}') + 1
        prediction = json.loads(text[start:end])
    except Exception:
        prediction = {
            "predicted_winner": teams[0] if teams else "TBD",
            "confidence": 50,
            "reasoning": "Insufficient data for prediction"
        }

    result = {
        "match_id": featured.get("id"),
        "name": featured.get("name"),
        "teams": teams,
        "status": status,
        "score": score,
        "matchStarted": featured.get("matchStarted"),
        "matchEnded": featured.get("matchEnded"),
        "venue": featured.get("venue"),
        "ai_prediction": prediction,
    }
    cache_set(cache_key, result, 60)
    return result