"""
Employee Attendance & Analytics API  (FastAPI + MongoDB)

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB   (a local .env is loaded for convenience; real environment variables win)

Layout of this file
  1. configuration and database handles
  2. small helpers: time conversion, rounding, shift maths, the business rules R1-R5
  3. request validation types and models
  4. endpoints: system, employees, attendance
  5. analytics: the aggregation pipelines and their endpoints
  6. admin: explain
"""
import calendar
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import date as Date
from datetime import datetime, time as Time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any, Literal, Optional

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.exceptions import RequestValidationError
from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    StringConstraints,
    model_validator,
)
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import AutoReconnect, DuplicateKeyError, PyMongoError

# --------------------------------------------------------------------------- #
# 1. Configuration
# --------------------------------------------------------------------------- #
load_dotenv()  # does not override variables that are already set in the environment

log = logging.getLogger("attendance")

# tz_aware=True: PyMongo returns UTC-aware datetimes instead of naive ones, so
# subtracting / comparing them with our own aware datetimes can never raise or drift.
client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

PRESENCE = ("PRESENT", "WFH", "ON_DUTY")  # R6
NO_PUNCH = ("ABSENT", "LEAVE")
Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]

LATE_GRACE = timedelta(minutes=10)  # R2
MIN_OVERTIME_MINUTES = 30  # R3
HALF_DAY_BELOW = Decimal("4.50")  # R5
MAX_SHIFT_SPAN = timedelta(hours=24)  # punch-out / correction window
MAX_TREND_DAYS = 92

MIN_MS, MAX_MS = 100_000_000_000, 4_102_444_800_000  # contract range for epoch milliseconds


# --------------------------------------------------------------------------- #
# Indexes (created at startup, idempotent)
# --------------------------------------------------------------------------- #
_indexes_ready = False


def ensure_indexes() -> None:
    """Create every index the service relies on. Safe to call repeatedly."""
    global _indexes_ready
    wanted = [
        # employees
        ("employees", [("emp_code", 1)], {"unique": True, "name": "emp_code_unique"}),
        ("employees", [("department", 1), ("emp_code", 1)], {"name": "dept_emp_code"}),
        ("employees", [("department", 1), ("joined_on", 1)], {"name": "dept_joined_on"}),
        ("employees", [("joined_on", 1)], {"name": "joined_on"}),
        # attendance_logs
        ("attendance_logs", [("emp_code", 1), ("date", 1)], {"unique": True, "name": "emp_date_unique"}),
        ("attendance_logs", [("date", -1), ("emp_code", 1)], {"name": "date_desc_emp"}),
        ("attendance_logs", [("status", 1), ("date", -1), ("emp_code", 1)], {"name": "status_date_emp"}),
        ("attendance_logs", [("emp_code", 1), ("punch_in", -1)], {"name": "emp_punch_in_desc"}),
    ]
    ok = True
    for coll, keys, opts in wanted:
        try:
            db[coll].create_index(keys, **opts)
        except AutoReconnect as exc:  # MongoDB unreachable: do not wait for every remaining index
            log.error("MongoDB not reachable while creating indexes: %s", exc)
            ok = False
            break
        except PyMongoError as exc:  # keep the service up; the log shows what is wrong
            ok = False
            log.error("could not create index %s on %s: %s", opts.get("name"), coll, exc)
    _indexes_ready = ok


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        ensure_indexes()
    except Exception:  # e.g. MongoDB not reachable yet; /health will report 503 until it is
        log.exception("index creation at startup failed")
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# 2. Helpers
# --------------------------------------------------------------------------- #
def to_ms(value: Any) -> Optional[int]:
    """BSON datetime -> integer epoch milliseconds (None stays None)."""
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:  # defensive: treat naive values as UTC, which is what MongoDB stores
        value = value.replace(tzinfo=UTC)
    return (value - EPOCH) // timedelta(milliseconds=1)


def from_ms(ms: int) -> datetime:
    """Epoch milliseconds -> UTC datetime truncated to whole seconds (R1)."""
    return EPOCH + timedelta(milliseconds=ms - ms % 1000)


def utc_now_seconds() -> datetime:
    return from_ms(to_ms(datetime.now(UTC)))


def as_aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def round_half_up(value: Any, places: int) -> float:
    """Round half away from zero on the decimal value (R4, R8). Never uses float rounding."""
    quantum = Decimal(1).scaleb(-places)
    number = value if isinstance(value, Decimal) else Decimal(str(value))
    return float(number.quantize(quantum, rounding=ROUND_HALF_UP))


def _hm(text: str) -> tuple[int, int]:
    hours, minutes = text.split(":")
    return int(hours), int(minutes)


def is_overnight(emp: dict) -> bool:
    return emp["shift_end"] <= emp["shift_start"]  # "HH:MM" strings compare like times


def attendance_date(punch_in: datetime, emp: dict) -> str:
    """R1: the IST calendar date a punch-in belongs to (previous day for early overnight punches)."""
    ist = punch_in.astimezone(IST)
    day = ist.date()
    if is_overnight(emp) and ist.time() < Time(*_hm(emp["shift_end"])):
        day -= timedelta(days=1)
    return day.isoformat()


def shift_window(date_str: str, emp: dict) -> tuple[datetime, datetime]:
    """Start and end instant of the shift that belongs to attendance date `date_str` (IST)."""
    day = Date.fromisoformat(date_str)
    sh, sm = _hm(emp["shift_start"])
    eh, em = _hm(emp["shift_end"])
    start = datetime(day.year, day.month, day.day, sh, sm, tzinfo=IST)
    end = datetime(day.year, day.month, day.day, eh, em, tzinfo=IST)
    if is_overnight(emp):
        end += timedelta(days=1)
    return start, end


def compute_derived(
    emp: dict, date_str: str, status: str, punch_in: Optional[datetime], punch_out: Optional[datetime]
) -> dict:
    """Rules R2-R5. Returns work_hours, late_minutes, overtime_minutes, half_day."""
    result = {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    if status not in PRESENCE or punch_in is None:
        return result

    start, end = shift_window(date_str, emp)

    after_start = punch_in - start
    if after_start > LATE_GRACE:  # R2: strictly more than 10:00
        result["late_minutes"] = after_start // timedelta(minutes=1)  # floor, counted from shift start

    if punch_out is not None:
        micros = (punch_out - punch_in) // timedelta(microseconds=1)
        hours = (Decimal(micros) / Decimal(3_600_000_000)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)  # R4
        result["work_hours"] = float(hours)
        result["half_day"] = hours < HALF_DAY_BELOW  # R5 uses the rounded value

        past_end = (punch_out - end) // timedelta(minutes=1)  # R3, floor
        if past_end >= MIN_OVERTIME_MINUTES:
            result["overtime_minutes"] = past_end
    return result


def record_out(doc: dict) -> dict:
    """Stored attendance document -> API shape (epoch ms, defaults for legacy documents)."""
    history = []
    for entry in doc.get("history") or []:
        changes = {}
        for field, pair in (entry.get("changes") or {}).items():
            convert = to_ms if field in ("punch_in", "punch_out") else (lambda v: v)
            changes[field] = {"from": convert(pair.get("from")), "to": convert(pair.get("to"))}
        history.append({"at": to_ms(entry.get("at")), "by": entry.get("by"), "reason": entry.get("reason"), "changes": changes})
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": to_ms(doc.get("punch_in")),
        "punch_out": to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes") or 0,
        "overtime_minutes": doc.get("overtime_minutes") or 0,
        "half_day": bool(doc.get("half_day", False)),
        "history": history,
    }


def employee_out(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": to_ms(doc.get("created_at")),
    }


def get_employee_or_404(emp_code: str) -> dict:
    emp = db.employees.find_one({"emp_code": emp_code}, {"_id": 0})
    if emp is None:
        raise HTTPException(404, "employee not found")
    return emp


def month_bounds(month: str) -> tuple[str, str]:
    year, mon = int(month[:4]), int(month[5:7])
    return f"{month}-01", f"{month}-{calendar.monthrange(year, mon)[1]:02d}"


def date_range(first: str, last: str) -> list[Date]:
    start = Date.fromisoformat(first)
    return [start + timedelta(days=i) for i in range((Date.fromisoformat(last) - start).days + 1)]


def count_weekdays(first: str, last: str) -> int:
    """Mon-Fri days between two ISO dates, inclusive (R7). 0 when last < first."""
    if last < first:
        return 0
    return sum(1 for d in date_range(first, last) if d.weekday() < 5)


def weekend_dates(first: str, last: str) -> list[str]:
    return [d.isoformat() for d in date_range(first, last) if d.weekday() >= 5]


# --------------------------------------------------------------------------- #
# 3. Validation types and models
# --------------------------------------------------------------------------- #
def _real_date(value: str) -> str:
    try:
        Date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("not a valid calendar date") from exc
    return value


def _real_month(value: str) -> str:
    try:
        Date(int(value[:4]), int(value[5:7]), 1)
    except ValueError as exc:
        raise ValueError("not a valid month") from exc
    return value


DateStr = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$"), AfterValidator(_real_date)]
MonthStr = Annotated[str, StringConstraints(pattern=r"^\d{4}-(0[1-9]|1[0-2])$"), AfterValidator(_real_month)]
HHMM = Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]
# strict: floats, strings and booleans are rejected; the range rejects "seconds"-sized numbers
EpochMs = Annotated[int, Field(strict=True, ge=MIN_MS, le=MAX_MS)]


class EmployeeIn(BaseModel):
    emp_code: Annotated[str, StringConstraints(pattern=r"^EMP\d{4,6}$")]
    name: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    email: Annotated[str, StringConstraints(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]
    department: Annotated[str, StringConstraints(min_length=1, max_length=50)]
    shift_start: HHMM = "09:30"
    shift_end: HHMM = "18:30"
    joined_on: DateStr

    @model_validator(mode="after")
    def _shift_times_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None
    status: PresenceStatus = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None


class RegularizeIn(BaseModel):
    status: Optional[Status] = None
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: Annotated[str, StringConstraints(min_length=5, max_length=200)]
    regularized_by: Annotated[str, StringConstraints(min_length=1, max_length=50)]

    @model_validator(mode="before")
    @classmethod
    def _no_explicit_nulls(cls, data):
        # An omitted field means "leave unchanged"; an explicit null is not a valid value.
        if isinstance(data, dict):
            for key in ("status", "punch_in", "punch_out"):
                if key in data and data[key] is None:
                    raise ValueError(f"{key} must not be null")
        return data


def missing_param(name: str) -> RequestValidationError:
    return RequestValidationError([{"type": "missing", "loc": ("query", name), "msg": "Field required", "input": None}])


# --------------------------------------------------------------------------- #
# 4. Endpoints: system, employees, attendance
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        db.command("ping")
    except PyMongoError:
        raise HTTPException(503, "database unavailable")
    if not _indexes_ready:  # MongoDB was down at startup: finish the job now
        ensure_indexes()
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = datetime.now(UTC)
    try:
        db.employees.insert_one(doc)  # the unique index on emp_code decides, not a prior find
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return employee_out(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    query = {"department": department} if department is not None else {}
    total = db.employees.count_documents(query)
    cursor = db.employees.find(query, {"_id": 0}).sort("emp_code", 1).skip((page - 1) * page_size).limit(page_size)
    return {"items": [employee_out(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = get_employee_or_404(body.emp_code)
    ts = from_ms(body.punched_at) if body.punched_at is not None else utc_now_seconds()
    day = attendance_date(ts, emp)
    doc = {
        "emp_code": body.emp_code,
        "date": day,
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        **compute_derived(emp, day, body.status, ts, None),
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)  # unique (emp_code, date) index makes this race-safe
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return record_out(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = get_employee_or_404(body.emp_code)
    out = from_ms(body.punched_at) if body.punched_at is not None else utc_now_seconds()

    # most recent record whose punch_in is at or before the punch-out (works across midnight)
    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": out}}, sort=[("punch_in", -1)]
    )
    if rec is None:
        raise HTTPException(404, "no punch-in found for this employee")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")

    p_in = as_aware(rec["punch_in"])
    if out <= p_in:
        raise HTTPException(422, "punched_at must be after the punch-in")
    if out - p_in > MAX_SHIFT_SPAN:
        raise HTTPException(422, "punched_at is more than 24 hours after the punch-in")

    derived = compute_derived(emp, rec["date"], rec["status"], p_in, out)
    # Compare-and-set: only succeeds while the record is still open and unchanged, so of two
    # simultaneous punch-outs exactly one matches.
    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None, "punch_in": rec["punch_in"], "status": rec["status"]},
        {"$set": {"punch_out": out, **derived}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "record was changed or already punched out")
    return record_out(updated)


def attendance_find_spec(
    emp_code: Optional[str], date_from: Optional[str], date_to: Optional[str], status: Optional[str], page: int, page_size: int
) -> dict:
    """Filter / sort / paging of GET /attendance. Shared with the explain endpoint."""
    query: dict = {}
    if emp_code:
        query["emp_code"] = emp_code
    if date_from or date_to:
        query["date"] = {}
        if date_from:
            query["date"]["$gte"] = date_from
        if date_to:
            query["date"]["$lte"] = date_to
    if status:
        query["status"] = status
    # With a single employee the date is unique, so sorting on date alone is the same order and
    # lets MongoDB walk the (emp_code, date) index backwards without a sort stage.
    sort = {"date": -1} if emp_code else {"date": -1, "emp_code": 1}
    return {"filter": query, "sort": sort, "skip": (page - 1) * page_size, "limit": page_size}


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[DateStr] = None,
    date_to: Optional[DateStr] = None,
    status: Optional[Status] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "date_from must not be after date_to")
    spec = attendance_find_spec(emp_code, date_from, date_to, status, page, page_size)
    total = db.attendance_logs.count_documents(spec["filter"])
    cursor = (
        db.attendance_logs.find(spec["filter"], {"_id": 0})
        .sort(list(spec["sort"].items()))
        .skip(spec["skip"])
        .limit(spec["limit"])
    )
    return {"items": [record_out(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, date: Annotated[DateStr, Path()], body: RegularizeIn):
    emp = get_employee_or_404(emp_code)
    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if rec is None:
        raise HTTPException(404, "no attendance record for that date")

    cur_in, cur_out = as_aware(rec.get("punch_in")), as_aware(rec.get("punch_out"))
    new_status = body.status or rec["status"]

    if new_status in NO_PUNCH:
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(422, f"status {new_status} clears the punch times; do not send punch_in / punch_out")
        new_in = new_out = None
    else:
        new_in = from_ms(body.punch_in) if body.punch_in is not None else cur_in
        new_out = from_ms(body.punch_out) if body.punch_out is not None else cur_out
        if new_in is None:
            raise HTTPException(422, f"status {new_status} requires a punch_in")
        if body.punch_in is not None and attendance_date(new_in, emp) != date:
            raise HTTPException(422, "punch_in must stay on the record's attendance date")
        if new_out is not None and (body.punch_in is not None or body.punch_out is not None):
            if new_out <= new_in:
                raise HTTPException(422, "punch_out must be after punch_in")
            if new_out - new_in > MAX_SHIFT_SPAN:
                raise HTTPException(422, "punch_out must be within 24 hours of punch_in")

    new = {"status": new_status, "punch_in": new_in, "punch_out": new_out}
    new.update(compute_derived(emp, date, new_status, new_in, new_out))
    old = {
        "status": rec["status"],
        "punch_in": cur_in,
        "punch_out": cur_out,
        "work_hours": rec.get("work_hours"),
        "late_minutes": rec.get("late_minutes") or 0,
        "overtime_minutes": rec.get("overtime_minutes") or 0,
        "half_day": bool(rec.get("half_day", False)),
    }
    changes = {field: {"from": old[field], "to": new[field]} for field in new if old[field] != new[field]}
    if not changes:
        raise HTTPException(422, "the request changes nothing")

    entry = {"at": datetime.now(UTC), "by": body.regularized_by, "reason": body.reason, "changes": changes}

    # Optimistic concurrency: the write only applies if the record is exactly as we read it
    # (same history length, same times and status). Otherwise someone else changed it first and
    # we answer 409 instead of silently overwriting their audit entry.
    history = rec.get("history")
    guard = {
        "_id": rec["_id"],
        "status": rec["status"],
        "punch_in": rec.get("punch_in"),
        "punch_out": rec.get("punch_out"),
        "history": {"$size": len(history)} if isinstance(history, list) else {"$exists": False},
    }
    updated = db.attendance_logs.find_one_and_update(
        guard, {"$set": new, "$push": {"history": entry}}, return_document=ReturnDocument.AFTER
    )
    if updated is None:
        raise HTTPException(409, "the record was modified concurrently; reload and retry")
    return record_out(updated)


# --------------------------------------------------------------------------- #
# 5. Analytics (MongoDB aggregation pipelines)
# --------------------------------------------------------------------------- #
def _presence_expr() -> dict:
    return {"$in": ["$status", list(PRESENCE)]}


def _credit_expr() -> dict:
    """A present record is worth 1 day, a half day 0.5. Missing half_day means false."""
    return {"$cond": [{"$eq": [{"$ifNull": ["$half_day", False]}, True]}, 0.5, 1]}


def _is_late_expr() -> dict:
    return {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}


def employee_monthly_pipeline(emp_code: str, first: str, last: str) -> list[dict]:
    weekend = weekend_dates(first, last)
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first, "$lte": last}}},
        {
            "$group": {
                "_id": None,
                "present_days": {
                    "$sum": {
                        "$cond": [{"$and": [_presence_expr(), {"$not": [{"$in": ["$date", weekend]}]}]}, _credit_expr(), 0]
                    }
                },
                "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
                "late_count": {"$sum": {"$cond": [_is_late_expr(), 1, 0]}},
                "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
            }
        },
    ]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: Annotated[MonthStr, Query()]):
    emp = get_employee_or_404(emp_code)
    first, last = month_bounds(month)
    working_days = count_weekdays(max(first, emp["joined_on"]), last)  # R7: counted from joined_on
    rows = list(db.attendance_logs.aggregate(employee_monthly_pipeline(emp_code, first, last)))
    s = rows[0] if rows else {}
    present = Decimal(str(s.get("present_days", 0)))
    pct = None if working_days == 0 else round_half_up(present * 100 / Decimal(working_days), 2)
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": round_half_up(present, 2),
        "leave_days": int(s.get("leave_days", 0)),
        "late_count": int(s.get("late_count", 0)),
        "total_late_minutes": int(s.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(s.get("total_overtime_minutes", 0)),
        "attendance_pct": pct,
    }


def department_summary_pipeline(first: str, last: str, department: Optional[str]) -> list[dict]:
    """Runs on `employees` so that people with zero logs still count (R9), and joins their month of logs."""
    weekend = weekend_dates(first, last)
    has_hours = {
        "$and": [_presence_expr(), {"$in": [{"$type": "$work_hours"}, ["double", "int", "long", "decimal"]]}]
    }
    match: dict = {"joined_on": {"$lte": last}}
    if department is not None:
        match["department"] = department
    return [
        {"$match": match},
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "pipeline": [
                    {"$match": {"date": {"$gte": first, "$lte": last}}},
                    {
                        "$group": {
                            "_id": None,
                            "present_days": {
                                "$sum": {
                                    "$cond": [
                                        {"$and": [_presence_expr(), {"$not": [{"$in": ["$date", weekend]}]}]},
                                        _credit_expr(),
                                        0,
                                    ]
                                }
                            },
                            # work hours are summed as whole hundredths so the later average is exact
                            "wh_cents": {
                                "$sum": {
                                    "$cond": [has_hours, {"$toLong": {"$round": [{"$multiply": ["$work_hours", 100]}, 0]}}, 0]
                                }
                            },
                            "wh_n": {"$sum": {"$cond": [has_hours, 1, 0]}},
                            "late_count": {"$sum": {"$cond": [_is_late_expr(), 1, 0]}},
                            "late_total": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                            "leave_count": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
                            "on_duty_count": {"$sum": {"$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]}},
                        }
                    },
                ],
                "as": "stats",
            }
        },
        {"$set": {"st": {"$arrayElemAt": ["$stats", 0]}}},  # missing for employees with no logs
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                "present_days": {"$sum": {"$ifNull": ["$st.present_days", 0]}},
                "wh_cents": {"$sum": {"$ifNull": ["$st.wh_cents", 0]}},
                "wh_n": {"$sum": {"$ifNull": ["$st.wh_n", 0]}},
                "late_count": {"$sum": {"$ifNull": ["$st.late_count", 0]}},
                "late_total": {"$sum": {"$ifNull": ["$st.late_total", 0]}},
                "leave_count": {"$sum": {"$ifNull": ["$st.leave_count", 0]}},
                "on_duty_count": {"$sum": {"$ifNull": ["$st.on_duty_count", 0]}},
            }
        },
        {"$sort": {"_id": 1}},
    ]


@app.get("/analytics/departments/summary")
def department_summary(month: Annotated[MonthStr, Query()], department: Optional[str] = None):
    first, last = month_bounds(month)
    items = []
    for g in db.employees.aggregate(department_summary_pipeline(first, last, department)):
        avg = None if g["wh_n"] == 0 else round_half_up(Decimal(g["wh_cents"]) / Decimal(g["wh_n"]) / 100, 2)
        items.append(
            {
                "department": g["_id"],
                "headcount": int(g["headcount"]),
                "present_days": round_half_up(g["present_days"], 2),
                "avg_work_hours": avg,
                "late_count": int(g["late_count"]),
                "total_late_minutes": int(g["late_total"]),
                "leave_count": int(g["leave_count"]),
                "on_duty_count": int(g["on_duty_count"]),
            }
        )
    return {"month": month, "items": items}


def late_leaderboard_pipeline(first: str, last: str, limit: int, department: Optional[str]) -> list[dict]:
    pipeline: list[dict] = [
        {"$match": {"date": {"$gte": first, "$lte": last}, "late_minutes": {"$gt": 0}}},
        {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": "$late_minutes"}, "late_count": {"$sum": 1}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "emp"}},
        {"$unwind": "$emp"},  # logs of unknown employees are dropped here
    ]
    if department is not None:
        pipeline.append({"$match": {"emp.department": department}})  # rank inside the department only
    pipeline += [
        # $rank is standard competition ranking: ties share a rank, the next rank is skipped
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},  # limit applies after ranking
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": "$emp.name",
                "department": "$emp.department",
                "total_late_minutes": 1,
                "late_count": 1,
            }
        },
    ]
    return pipeline


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: Annotated[MonthStr, Query()], limit: int = Query(10, ge=1, le=50), department: Optional[str] = None
):
    first, last = month_bounds(month)
    items = [
        {
            "rank": int(r["rank"]),
            "emp_code": r["emp_code"],
            "name": r["name"],
            "department": r["department"],
            "total_late_minutes": int(r["total_late_minutes"]),
            "late_count": int(r["late_count"]),
        }
        for r in db.attendance_logs.aggregate(late_leaderboard_pipeline(first, last, limit, department))
    ]
    return {"month": month, "items": items}


def check_trend_range(frm: str, to: str) -> None:
    if to < frm:
        raise HTTPException(422, "'to' must not be before 'from'")
    if (Date.fromisoformat(to) - Date.fromisoformat(frm)).days + 1 > MAX_TREND_DAYS:
        raise HTTPException(422, f"the range must not exceed {MAX_TREND_DAYS} days")


def department_trend_pipeline(department: str, frm: str, to: str) -> list[dict]:
    """
    One row per calendar day, built inside MongoDB:
      employees (this department) -> per-employee daily sums via $lookup -> $facet splits
      "who had joined" (for headcount) from "sums per day" -> $range/$map lays out every day of
      the range (gaps included) -> $setWindowFields adds the 7-row moving average.
    Rates are carried as whole ten-thousandths so the half-up rounding is exact integer maths.
    """
    start = datetime.fromisoformat(frm).replace(tzinfo=UTC)  # only used as a day counter
    days = (Date.fromisoformat(to) - Date.fromisoformat(frm)).days + 1
    day_at = {"$add": [start, {"$multiply": ["$$i", 86_400_000]}]}

    present_halves = {"$multiply": [2, {"$ifNull": ["$$pd.present", 0]}]}  # present_count is a multiple of 0.5
    rate_t = {  # floor((10000 * present / headcount) + 1/2), written with integers only
        "$floor": {
            "$divide": [
                {"$add": [{"$multiply": [10000, present_halves]}, "$$hc"]},
                {"$multiply": [2, "$$hc"]},
            ]
        }
    }
    has_rate = {"$and": ["$$wd", {"$gt": ["$$hc", 0]}]}

    day_row = {
        "$let": {
            "vars": {
                "d": {"$dateToString": {"format": "%Y-%m-%d", "date": day_at}},
                "dow": {"$dayOfWeek": day_at},  # 1 = Sunday ... 7 = Saturday
            },
            "in": {
                "$let": {
                    "vars": {
                        "pd": {
                            "$arrayElemAt": [
                                {"$filter": {"input": "$perday", "as": "p", "cond": {"$eq": ["$$p._id", "$$d"]}}},
                                0,
                            ]
                        },
                        "hc": {
                            "$size": {
                                "$filter": {
                                    "input": {"$ifNull": [{"$arrayElemAt": ["$joined.j", 0]}, []]},
                                    "as": "j",
                                    "cond": {"$lte": ["$$j", "$$d"]},
                                }
                            }
                        },
                        "wd": {"$and": [{"$gte": ["$$dow", 2]}, {"$lte": ["$$dow", 6]}]},
                    },
                    "in": {
                        "date": "$$d",
                        "is_working_day": "$$wd",
                        "headcount": "$$hc",
                        "present_count": {"$ifNull": ["$$pd.present", 0]},
                        "late_count": {"$ifNull": ["$$pd.late", 0]},
                        "has_rate": {"$cond": [has_rate, 1, 0]},
                        "rate_t": {"$cond": [has_rate, rate_t, None]},
                    },
                }
            },
        }
    }

    window = {"documents": [-6, 0]}  # current row and up to 6 before it, never before `from`
    return [
        {"$match": {"department": department}},
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "pipeline": [
                    {"$match": {"date": {"$gte": frm, "$lte": to}}},
                    {
                        "$group": {
                            "_id": "$date",
                            "present": {"$sum": {"$cond": [_presence_expr(), _credit_expr(), 0]}},
                            "late": {"$sum": {"$cond": [_is_late_expr(), 1, 0]}},
                        }
                    },
                ],
                "as": "days",
            }
        },
        {"$project": {"_id": 0, "joined_on": 1, "days": 1}},
        {
            "$facet": {
                "joined": [{"$group": {"_id": None, "j": {"$push": "$joined_on"}}}],
                "perday": [
                    {"$unwind": "$days"},
                    {
                        "$group": {
                            "_id": "$days._id",
                            "present": {"$sum": "$days.present"},
                            "late": {"$sum": "$days.late"},
                        }
                    },
                ],
            }
        },
        {"$project": {"rows": {"$map": {"input": {"$range": [0, days]}, "as": "i", "in": day_row}}}},
        {"$unwind": "$rows"},
        {"$replaceRoot": {"newRoot": "$rows"}},
        {
            "$setWindowFields": {
                "sortBy": {"date": 1},
                "output": {
                    "rate_sum": {"$sum": "$rate_t", "window": window},  # nulls are ignored
                    "rate_n": {"$sum": "$has_rate", "window": window},
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "is_working_day": 1,
                "headcount": 1,
                "present_count": 1,
                "late_count": 1,
                "attendance_rate": {"$cond": [{"$eq": ["$has_rate", 1]}, {"$divide": ["$rate_t", 10000]}, None]},
                "moving_avg_7d": {
                    "$cond": [
                        {"$eq": ["$rate_n", 0]},
                        None,
                        {  # half-up mean of the window, still in ten-thousandths
                            "$divide": [
                                {
                                    "$floor": {
                                        "$divide": [
                                            {"$add": [{"$multiply": [2, "$rate_sum"]}, "$rate_n"]},
                                            {"$multiply": [2, "$rate_n"]},
                                        ]
                                    }
                                },
                                10000,
                            ]
                        },
                    ]
                },
            }
        },
        {"$sort": {"date": 1}},
    ]


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str, frm: Annotated[DateStr, Query(alias="from")], to: Annotated[DateStr, Query()]
):
    check_trend_range(frm, to)
    if db.employees.find_one({"department": department}, {"_id": 1}) is None:
        raise HTTPException(404, "unknown department")
    items = []
    for r in db.employees.aggregate(department_trend_pipeline(department, frm, to)):
        items.append(
            {
                "date": r["date"],
                "is_working_day": bool(r["is_working_day"]),
                "headcount": int(r["headcount"]),
                "present_count": r["present_count"],
                "late_count": int(r["late_count"]),
                "attendance_rate": r.get("attendance_rate"),
                "moving_avg_7d": r.get("moving_avg_7d"),
            }
        )
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# 6. Admin: explain
# --------------------------------------------------------------------------- #
def _explain(command: dict) -> dict:
    result = db.command({"explain": command, "verbosity": "executionStats"})
    return json.loads(json_util.dumps(result, json_options=json_util.RELAXED_JSON_OPTIONS))


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: Literal["attendance_list", "employee_monthly", "department_summary", "late_leaderboard", "department_trend"],
    emp_code: Optional[str] = None,
    month: Optional[MonthStr] = None,
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[DateStr] = None,
    date_to: Optional[DateStr] = None,
    status: Optional[Status] = None,
    frm: Annotated[Optional[DateStr], Query(alias="from")] = None,
    to: Optional[DateStr] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if endpoint == "attendance_list":
        if date_from and date_to and date_from > date_to:
            raise HTTPException(422, "date_from must not be after date_to")
        spec = attendance_find_spec(emp_code, date_from, date_to, status, page, page_size)
        collection, command = "attendance_logs", {"find": "attendance_logs", **spec}
    elif endpoint == "employee_monthly":
        if not emp_code:
            raise missing_param("emp_code")
        if not month:
            raise missing_param("month")
        first, last = month_bounds(month)
        collection = "attendance_logs"
        command = {"aggregate": collection, "pipeline": employee_monthly_pipeline(emp_code, first, last), "cursor": {}}
    elif endpoint == "department_summary":
        if not month:
            raise missing_param("month")
        first, last = month_bounds(month)
        collection = "employees"
        command = {"aggregate": collection, "pipeline": department_summary_pipeline(first, last, department), "cursor": {}}
    elif endpoint == "late_leaderboard":
        if not month:
            raise missing_param("month")
        first, last = month_bounds(month)
        collection = "attendance_logs"
        command = {"aggregate": collection, "pipeline": late_leaderboard_pipeline(first, last, limit, department), "cursor": {}}
    else:  # department_trend
        if not department:
            raise missing_param("department")
        if not frm:
            raise missing_param("from")
        if not to:
            raise missing_param("to")
        check_trend_range(frm, to)
        collection = "employees"
        command = {"aggregate": collection, "pipeline": department_trend_pipeline(department, frm, to), "cursor": {}}
    return {"endpoint": endpoint, "collection": collection, "explain": _explain(command)}
