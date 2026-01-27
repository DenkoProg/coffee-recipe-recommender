from typing import List
from pydantic import BaseModel
from pathlib import Path

import csv

class UserOut(BaseModel):
    user_id: str
    username: str | None = None

def list_users(limit: int) -> List[UserOut]:
    USERS_CSV_PATH = Path("data/users.csv")
    if limit <= 0:
        return []

    if not USERS_CSV_PATH.exists():
        raise FileNotFoundError(f"{USERS_CSV_PATH} not found")

    users: List[UserOut] = []

    with USERS_CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Expect columns like: user_id, username (username optional)
        if not reader.fieldnames or "user_id" not in reader.fieldnames:
            raise ValueError("users.csv must contain a 'user_id' column (header row required)")

        for row in reader:
            user_id = (row.get("user_id") or "").strip()
            if not user_id:
                continue  # skip invalid rows

            username_raw = row.get("username")
            username = username_raw.strip() if isinstance(username_raw, str) and username_raw.strip() else None

            users.append(UserOut(user_id=user_id, username=username))

            if len(users) >= limit:
                break

    return users