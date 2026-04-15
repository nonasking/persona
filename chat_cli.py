"""
로컬 Persona 서버와 대화하는 CLI 스크립트.

사용법:
    poetry run python chat_cli.py
"""

import json
import sys
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000/api"


def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}") as resp:
        return json.loads(resp.read())


def print_profile_summary() -> None:
    try:
        profile = get("/profile/")
    except Exception:
        return
    traits = profile.get("traits", {})
    print("\n--- 현재 프로필 ---")
    for field in ("formality", "directness", "humor", "empathy", "verbosity"):
        t = traits.get(field, {})
        score = t.get("score", 0.5)
        obs = t.get("observations", 0)
        bar = "#" * int(score * 20)
        print(f"  {field:<12} {score:.2f}  [{bar:<20}]  (관찰 {obs}회)")
    phrases = [p["phrase"] for p in traits.get("characteristic_phrases", [])[:5]]
    if phrases:
        print(f"  phrases    : {', '.join(phrases)}")
    topics = list((traits.get("topics") or {}).keys())[:5]
    if topics:
        print(f"  topics     : {', '.join(topics)}")
    print(f"  학습 횟수   : {profile.get('update_count', 0)}회")
    print("------------------\n")


def main() -> None:
    print("Persona Chat — 종료하려면 Ctrl+C 또는 'quit' 입력")
    print("프로필을 보려면 '/profile' 입력\n")

    session_id: int | None = None

    while True:
        try:
            user_input = input("나  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "종료"):
            print("종료합니다.")
            sys.exit(0)

        if user_input == "/profile":
            print_profile_summary()
            continue

        body: dict = {"message": user_input}
        if session_id is not None:
            body["session_id"] = session_id

        try:
            resp = post("/chat/", body)
        except urllib.error.URLError:
            print("[오류] 서버에 연결할 수 없습니다. 'poetry run python manage.py runserver' 가 실행 중인지 확인하세요.\n")
            continue
        except Exception as e:
            print(f"[오류] {e}\n")
            continue

        session_id = resp["session_id"]
        print(f"AI  > {resp['reply']}\n")


if __name__ == "__main__":
    main()
