from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, conint, confloat


class UsersOut(BaseModel):
    user_id: str
    username: str | None = None

class MetaOut(BaseModel):
    equipment: list[str]
    products: list[str]

def list_users(limit: int) -> list[UserOut]:
    USERS_CSV_PATH = Path("data/users.csv")
    if limit <= 0:
        return []

    if not USERS_CSV_PATH.exists():
        raise FileNotFoundError(f"{USERS_CSV_PATH} not found")

    users: list[UserOut] = []

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

def get_meta_from_dataset(users_csv_path: str | Path = Path("data/users.csv")) -> dict:
    equipment = set()
    products = set()

    with open(users_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("owned_equipment"):
                for x in json.loads(row["owned_equipment"]):
                    equipment.add(str(x))

            if row.get("available_products"):
                for x in json.loads(row["available_products"]):
                    products.add(str(x))

    return {
        "equipment": sorted(equipment),
        "products": sorted(products),
    }

def _parse_json_array(cell: str | None) -> list[str]:
    if not cell:
        return []
    s = cell.strip()
    if not s or s.lower() in {"null", "none", "nan"}:
        return []
    try:
        val = json.loads(s)
    except json.JSONDecodeError:
        return []
    if not isinstance(val, list):
        return []
    out: list[str] = []
    for x in val:
        if x is None:
            continue
        t = str(x).strip()
        if t:
            out.append(t)
    return out


def _parse_float(cell: str | None) -> float | None:
    if cell is None:
        return None
    s = cell.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_int(cell: str | None) -> int | None:
    if cell is None:
        return None
    s = cell.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


class UserOut(BaseModel):
    user_id: str
    username: str | None = None

    owned_equipment: list[str] = Field(default_factory=list)
    available_products: list[str] = Field(default_factory=list)

    taste_pref_bitterness: float | None = None
    taste_pref_sweetness: float | None = None
    taste_pref_acidity: float | None = None
    taste_pref_body: float | None = None

    preferred_strength: int | None = None
    preferred_portion_size: Literal["small", "medium", "large"] | None = None

    dietary_restrictions: list[str] = Field(default_factory=list)

    # Keep as string to match your example exactly; CSV can store "Z" timestamps easily.
    account_created: str | None = None


def get_user_from_dataset(user_id: str, users_csv_path: str | Path = Path("data/users.csv")) -> UserOut:
    """
    Return a single user row by user_id from users.csv (first match).
    Raises ValueError if not found.
    """
    path = Path(users_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"users.csv not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("user_id") or "").strip() != user_id:
                continue

            return UserOut(
                user_id=user_id,
                username=(row.get("username") or None),

                owned_equipment=_parse_json_array(row.get("owned_equipment")),
                available_products=_parse_json_array(row.get("available_products")),

                taste_pref_bitterness=_parse_float(row.get("taste_pref_bitterness")),
                taste_pref_sweetness=_parse_float(row.get("taste_pref_sweetness")),
                taste_pref_acidity=_parse_float(row.get("taste_pref_acidity")),
                taste_pref_body=_parse_float(row.get("taste_pref_body")),

                preferred_strength=_parse_int(row.get("preferred_strength")),
                preferred_portion_size=(row.get("preferred_portion_size") or None),

                dietary_restrictions=_parse_json_array(row.get("dietary_restrictions")),

                account_created=(row.get("account_created") or None),
            )

    raise ValueError(f"user_id not found: {user_id}")

class UserCreate(BaseModel):
    username: str | None = None
    owned_equipment: list[str] = Field(default_factory=list)
    available_products: list[str] = Field(default_factory=list)
    taste_pref_bitterness: confloat(ge=0.0, le=1.0) = 0.0
    taste_pref_sweetness: confloat(ge=0.0, le=1.0) = 0.0
    taste_pref_acidity: confloat(ge=0.0, le=1.0) = 0.0
    taste_pref_body: confloat(ge=0.0, le=1.0) = 0.0
    preferred_strength: conint(ge=1, le=5) = 3
    preferred_portion_size: str = "medium"
    dietary_restrictions: list[str] = Field(default_factory=list)

def _ensure_csv_with_header(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def _read_all_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def _next_user_id(rows: list[dict[str, str]]) -> str:
    max_n = -1
    for row in rows:
        uid = (row.get("user_id") or "").strip()
        # expects "user_00012"
        if uid.startswith("user_"):
            tail = uid[5:]
            if tail.isdigit():
                max_n = max(max_n, int(tail))
    return f"user_{max_n + 1:05d}"


def create_user_in_dataset(payload: UserCreate, users_csv_path: str | Path = Path("data/users.csv")) -> None:
    fieldnames = [
        "user_id",
        "username",
        "owned_equipment",
        "available_products",
        "taste_pref_bitterness",
        "taste_pref_sweetness",
        "taste_pref_acidity",
        "taste_pref_body",
        "preferred_strength",
        "preferred_portion_size",
        "dietary_restrictions",
        "account_created",
    ]
    _ensure_csv_with_header(users_csv_path, fieldnames)

    rows = _read_all_rows(users_csv_path)
    user_id = _next_user_id(rows)
    account_created = datetime.utcnow().replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")

    row = {
        "user_id": user_id,
        "username": payload.username or "",
        "owned_equipment": json.dumps(payload.owned_equipment, ensure_ascii=False),
        "available_products": json.dumps(payload.available_products, ensure_ascii=False),
        "taste_pref_bitterness": str(float(payload.taste_pref_bitterness)),
        "taste_pref_sweetness": str(float(payload.taste_pref_sweetness)),
        "taste_pref_acidity": str(float(payload.taste_pref_acidity)),
        "taste_pref_body": str(float(payload.taste_pref_body)),
        "preferred_strength": str(int(payload.preferred_strength)),
        "preferred_portion_size": payload.preferred_portion_size,
        "dietary_restrictions": json.dumps(payload.dietary_restrictions, ensure_ascii=False),
        "account_created": account_created,
    }

    # append
    with users_csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)

    return UserOut(
        user_id=user_id,
        username=payload.username,
        owned_equipment=payload.owned_equipment,
        available_products=payload.available_products,
        taste_pref_bitterness=float(payload.taste_pref_bitterness),
        taste_pref_sweetness=float(payload.taste_pref_sweetness),
        taste_pref_acidity=float(payload.taste_pref_acidity),
        taste_pref_body=float(payload.taste_pref_body),
        preferred_strength=int(payload.preferred_strength),
        preferred_portion_size=payload.preferred_portion_size,
        dietary_restrictions=payload.dietary_restrictions,
        account_created=account_created,
    )