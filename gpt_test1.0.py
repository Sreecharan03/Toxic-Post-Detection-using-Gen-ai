import os
import sys
import json
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

TWEET_ID = "2001161855340003510"  # from your URL

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(level, msg):
    print(f"[{ts()}] {level:<5} | {msg}")

def fail(msg, code=1):
    log("ERROR", msg)
    sys.exit(code)

def main():
    load_dotenv("/teamspace/studios/this_studio/.env")

    bearer = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer:
        fail("TWITTER_BEARER_TOKEN missing in env.")

    # If user accidentally stored "Bearer xxx", normalize it:
    if bearer.lower().startswith("bearer "):
        bearer = bearer.split(" ", 1)[1].strip()
        log("WARN", "Bearer token had 'Bearer ' prefix in env; normalized it.")

    url = f"https://api.x.com/2/tweets/{TWEET_ID}"
    params = {
        "tweet.fields": "created_at,public_metrics,lang,conversation_id,entities,possibly_sensitive",
        "expansions": "author_id,attachments.media_keys",
        "user.fields": "name,username,profile_image_url,verified",
        "media.fields": "type,url,preview_image_url,alt_text",
    }
    headers = {"Authorization": f"Bearer {bearer}"}

    log("INFO", f"GET {url}")
    log("INFO", f"Params: {params}")

    t0 = time.time()
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
    except requests.RequestException as e:
        fail(f"Network/request error: {e}")

    dt = time.time() - t0
    log("INFO", f"Status: {r.status_code} (took {dt:.2f}s)")

    # Friendly error logs
    if r.status_code == 401:
        fail("401 Unauthorized: Bearer token invalid/expired OR still URL-encoded OR wrong app permissions.")
    if r.status_code == 403:
        fail("403 Forbidden: Your app/dev plan may not allow this endpoint or the tweet is not accessible.")
    if r.status_code == 429:
        reset = r.headers.get("x-rate-limit-reset")
        fail(f"429 Rate limit: Too many requests. Reset header: {reset}")
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text[:500]
        fail(f"HTTP {r.status_code} error response: {err}")

    # Success
    try:
        data = r.json()
    except Exception:
        fail("Response was not JSON. Raw: " + r.text[:500])

    log("OK", "Fetched tweet successfully.")
    print("\n=== RAW JSON ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    # Small human-readable summary
    tweet = data.get("data", {})
    includes = data.get("includes", {})
    users = {u["id"]: u for u in includes.get("users", [])}

    author = users.get(tweet.get("author_id"), {})
    metrics = tweet.get("public_metrics", {})

    print("\n=== SUMMARY ===")
    print(f"Author   : @{author.get('username','?')} ({author.get('name','?')})")
    print(f"Created  : {tweet.get('created_at','?')}")
    print(f"Text     : {tweet.get('text','?')}")
    print(f"Likes    : {metrics.get('like_count','?')}")
    print(f"Retweets : {metrics.get('retweet_count','?')}")
    print(f"Replies  : {metrics.get('reply_count','?')}")
    print(f"Quotes   : {metrics.get('quote_count','?')}")

if __name__ == "__main__":
    main()
