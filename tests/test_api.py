"""
Tests for app/main.py.

  pip install -r requirements-dev.txt
  pytest -q                                   # rules + CRUD on an in-memory mock (analytics/explain skipped)
  MONGO_URI=mongodb://localhost:27017 pytest -q   # everything, against a real MongoDB 6.0+
                                                  # (uses a throw-away database that is dropped afterwards)

The analytics tests build an awkward dataset (ties, weekend records, a mid-month joiner, an
overnight-shift employee, employees with no logs, logs of unknown employees, legacy documents)
and compare the API with a small, independent, plain-Python oracle.
"""
import os
import random
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

import pytest
from fastapi.testclient import TestClient

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))


def ms(year, month, day, hour=0, minute=0, second=0, tz=IST):
    return int(datetime(year, month, day, hour, minute, second, tzinfo=tz).timestamp()) * 1000


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _connect():
    uri = os.getenv("MONGO_URI")
    if uri:
        from pymongo import MongoClient

        client = MongoClient(uri, tz_aware=True, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        return client, client["attendance_test_" + uuid.uuid4().hex[:8]], True
    import mongomock

    client = mongomock.MongoClient(tz_aware=True)
    return client, client["attendance_test"], False


@pytest.fixture()
def env():
    import app.main as m

    client, database, real = _connect()
    saved = (m.client, m.db)
    m.client, m.db = client, database
    m._indexes_ready = False
    m.ensure_indexes()
    yield types.SimpleNamespace(m=m, db=database, real=real, http=TestClient(m.app, raise_server_exceptions=False))
    if real:
        client.drop_database(database.name)
    m.client, m.db = saved


needs_mongo = pytest.mark.skipif(not os.getenv("MONGO_URI"), reason="needs a real MongoDB (set MONGO_URI)")


def add_emp(http, code="EMP0001", dept="Engineering", start="09:30", end="18:30", joined="2026-01-05"):
    r = http.post(
        "/employees",
        json={"emp_code": code, "name": f"N {code}", "email": f"{code.lower()}@x.com", "department": dept,
              "shift_start": start, "shift_end": end, "joined_on": joined},
    )
    assert r.status_code == 201, r.text
    return r.json()


# --------------------------------------------------------------------------- #
# business rules (pure functions)
# --------------------------------------------------------------------------- #
EMP = {"shift_start": "09:30", "shift_end": "18:30"}
NIGHT = {"shift_start": "22:00", "shift_end": "06:00"}


def utc(*args):
    return datetime(*args, tzinfo=IST).astimezone(UTC)


def test_late_rule_r2(env):
    f = env.m.compute_derived
    day = "2026-07-06"
    assert f(EMP, day, "PRESENT", utc(2026, 7, 6, 9, 40, 0), None)["late_minutes"] == 0
    assert f(EMP, day, "PRESENT", utc(2026, 7, 6, 9, 40, 1), None)["late_minutes"] == 10
    assert f(EMP, day, "PRESENT", utc(2026, 7, 6, 10, 15, 59), None)["late_minutes"] == 45
    assert f(EMP, day, "PRESENT", utc(2026, 7, 6, 8, 0, 0), None)["late_minutes"] == 0


def test_overtime_rule_r3(env):
    f = env.m.compute_derived
    p_in = utc(2026, 7, 6, 9, 30, 0)
    assert f(EMP, "2026-07-06", "PRESENT", p_in, utc(2026, 7, 6, 18, 59, 59))["overtime_minutes"] == 0
    assert f(EMP, "2026-07-06", "PRESENT", p_in, utc(2026, 7, 6, 19, 0, 0))["overtime_minutes"] == 30
    assert f(EMP, "2026-07-06", "PRESENT", p_in, utc(2026, 7, 6, 19, 10, 59))["overtime_minutes"] == 40


def test_work_hours_half_up_r4_r5(env):
    f = env.m.compute_derived
    p_in = utc(2026, 7, 6, 9, 0, 0)
    # 8h 02m 42s = 8.045 h -> half-up 8.05 (Python round() gives 8.04)
    assert f(EMP, "2026-07-06", "PRESENT", p_in, p_in + timedelta(seconds=28962))["work_hours"] == 8.05
    # 4h 29m 42s = 4.495 h -> 4.50 -> NOT a half day
    r = f(EMP, "2026-07-06", "PRESENT", p_in, p_in + timedelta(hours=4, minutes=29, seconds=42))
    assert r["work_hours"] == 4.5 and r["half_day"] is False
    r = f(EMP, "2026-07-06", "PRESENT", p_in, p_in + timedelta(hours=4, minutes=29, seconds=41))
    assert r["work_hours"] == 4.49 and r["half_day"] is True


def test_overnight_r1(env):
    m = env.m
    assert m.attendance_date(utc(2026, 7, 6, 21, 55, 0), NIGHT) == "2026-07-06"
    assert m.attendance_date(utc(2026, 7, 7, 2, 0, 0), NIGHT) == "2026-07-06"  # before 06:00 -> previous day
    assert m.attendance_date(utc(2026, 7, 7, 6, 0, 0), NIGHT) == "2026-07-07"
    start, end = m.shift_window("2026-07-06", NIGHT)
    assert end - start == timedelta(hours=8)
    r = m.compute_derived(NIGHT, "2026-07-06", "PRESENT", utc(2026, 7, 6, 21, 55, 0), utc(2026, 7, 7, 6, 40, 0))
    assert (r["late_minutes"], r["overtime_minutes"], r["work_hours"]) == (0, 40, 8.75)


# --------------------------------------------------------------------------- #
# employees
# --------------------------------------------------------------------------- #
def test_employees_create_validate_list(env):
    h = env.http
    e = add_emp(h, "EMP0001")
    assert set(e) == {"emp_code", "name", "email", "department", "shift_start", "shift_end", "joined_on", "created_at"}
    assert isinstance(e["created_at"], int) and e["created_at"] > MIN_MS
    assert h.post("/employees", json={**e, "email": "a@b.co"}).status_code == 409
    base = {"emp_code": "EMP0002", "name": "n", "email": "a@b.co", "department": "d", "joined_on": "2026-01-01"}
    assert h.post("/employees", json=base).status_code == 201
    for bad in (
        {"emp_code": "E1"}, {"emp_code": "EMP12"}, {"email": "nope"}, {"joined_on": "2026-02-30"},
        {"joined_on": "2026-2-3"}, {"shift_start": "9:30"}, {"shift_end": "25:00"}, {"name": ""},
        {"shift_start": "09:30", "shift_end": "09:30"},
    ):
        assert h.post("/employees", json={**base, "emp_code": "EMP0003", **bad}).status_code == 422, bad
    for i in range(3, 8):
        add_emp(h, f"EMP000{i}", "Sales" if i % 2 else "Engineering")
    page = h.get("/employees?page=1&page_size=3").json()
    assert [x["emp_code"] for x in page["items"]] == ["EMP0001", "EMP0002", "EMP0003"] and page["total"] == 7
    page2 = h.get("/employees?page=2&page_size=3").json()
    assert [x["emp_code"] for x in page2["items"]] == ["EMP0004", "EMP0005", "EMP0006"]
    sales = h.get("/employees?department=Sales").json()
    assert sales["total"] == 3 and all(x["department"] == "Sales" for x in sales["items"])
    assert h.get("/employees?department=sales").json()["total"] == 0  # case-sensitive
    for q in ("page=0", "page_size=0", "page_size=101"):
        assert h.get("/employees?" + q).status_code == 422


MIN_MS = 100_000_000_000


# --------------------------------------------------------------------------- #
# punch-in / punch-out
# --------------------------------------------------------------------------- #
def test_punch_in_and_out_flow(env):
    h = env.http
    add_emp(h, "EMP0001")
    r = h.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 10, 15, 59) + 900})
    assert r.status_code == 201, r.text
    rec = r.json()
    assert rec["date"] == "2026-07-06" and rec["late_minutes"] == 45 and rec["status"] == "PRESENT"
    assert rec["punch_in"] == ms(2026, 7, 6, 10, 15, 59)  # truncated to whole seconds
    assert rec["punch_out"] is None and rec["work_hours"] is None and rec["history"] == []
    assert h.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 11)}).status_code == 409
    assert h.post("/attendance/punch-in", json={"emp_code": "EMP9999"}).status_code == 404

    out = h.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 19, 10, 59)})
    assert out.status_code == 200, out.text
    o = out.json()
    assert (o["work_hours"], o["overtime_minutes"], o["half_day"]) == (8.92, 40, False)
    assert h.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 20)}).status_code == 409
    assert h.post("/attendance/punch-out", json={"emp_code": "EMP9999"}).status_code == 404


def test_punch_validation(env):
    h = env.http
    add_emp(h, "EMP0001")
    for bad in (1783312500, 1783312500000.0, "1783312500000", True, 99_999_999_999, 4_102_444_800_001, -5):
        r = h.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": bad})
        assert r.status_code == 422, bad
    for st in ("ABSENT", "LEAVE", "nope"):
        assert h.post("/attendance/punch-in", json={"emp_code": "EMP0001", "status": st}).status_code == 422
    # derived fields sent by a client are ignored
    r = h.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 9, 0), "late_minutes": 99, "work_hours": 5})
    assert r.status_code == 201 and r.json()["late_minutes"] == 0 and r.json()["work_hours"] is None
    # punch-out: not after punch-in / more than 24h later
    assert h.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 9, 0)}).status_code == 422
    assert h.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 7, 9, 0, 1)}).status_code == 422
    assert h.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 8, 0)}).status_code == 404


def test_overnight_punch_across_midnight(env):
    h = env.http
    add_emp(h, "EMP0005", "Support", "22:00", "06:00", "2026-03-10")
    r = h.post("/attendance/punch-in", json={"emp_code": "EMP0005", "punched_at": ms(2026, 7, 6, 21, 55)})
    assert r.json()["date"] == "2026-07-06" and r.json()["late_minutes"] == 0
    o = h.post("/attendance/punch-out", json={"emp_code": "EMP0005", "punched_at": ms(2026, 7, 7, 6, 40)}).json()
    assert (o["work_hours"], o["overtime_minutes"]) == (8.75, 40)
    # an early-morning punch-in belongs to the previous day's shift
    r = h.post("/attendance/punch-in", json={"emp_code": "EMP0005", "punched_at": ms(2026, 7, 9, 2, 0)})
    assert r.json()["date"] == "2026-07-08"


def test_concurrent_punch_in_and_out(env):
    h = env.http
    add_emp(h, "EMP0001")
    body = {"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 9, 30)}
    with ThreadPoolExecutor(12) as ex:
        codes = sorted(f.result().status_code for f in [ex.submit(h.post, "/attendance/punch-in", json=body) for _ in range(12)])
    assert codes == [201] + [409] * 11
    body = {"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 18, 30)}
    with ThreadPoolExecutor(12) as ex:
        codes = sorted(f.result().status_code for f in [ex.submit(h.post, "/attendance/punch-out", json=body) for _ in range(12)])
    assert codes == [200] + [409] * 11


# --------------------------------------------------------------------------- #
# regularization (PATCH)
# --------------------------------------------------------------------------- #
def make_day(h, code="EMP0001", day=6, p_in=(10, 5), p_out=(19, 10)):
    h.post("/attendance/punch-in", json={"emp_code": code, "punched_at": ms(2026, 7, day, *p_in)})
    return h.post("/attendance/punch-out", json={"emp_code": code, "punched_at": ms(2026, 7, day, *p_out)}).json()


def patch(h, code, day, **body):
    return h.patch(f"/attendance/{code}/{day}", json={"reason": "biometric glitch", "regularized_by": "hr.admin", **body})


def test_regularize(env):
    h = env.http
    add_emp(h, "EMP0001")
    before = make_day(h)
    assert before["late_minutes"] == 35
    r = patch(h, "EMP0001", "2026-07-06", punch_in=ms(2026, 7, 6, 9, 28))
    assert r.status_code == 200, r.text
    rec = r.json()
    assert rec["late_minutes"] == 0 and rec["work_hours"] == 9.7 and rec["overtime_minutes"] == 40
    assert len(rec["history"]) == 1
    entry = rec["history"][0]
    assert entry["by"] == "hr.admin" and entry["reason"] == "biometric glitch" and isinstance(entry["at"], int)
    assert set(entry["changes"]) == {"punch_in", "late_minutes", "work_hours"}  # only what changed
    assert entry["changes"]["punch_in"] == {"from": ms(2026, 7, 6, 10, 5), "to": ms(2026, 7, 6, 9, 28)}
    assert entry["changes"]["late_minutes"] == {"from": 35, "to": 0}
    # same request again changes nothing
    assert patch(h, "EMP0001", "2026-07-06", punch_in=ms(2026, 7, 6, 9, 28)).status_code == 422
    # status change clears punches
    r = patch(h, "EMP0001", "2026-07-06", status="LEAVE").json()
    assert r["punch_in"] is None and r["punch_out"] is None and r["work_hours"] is None
    assert len(r["history"]) == 2 and r["history"][0] == rec["history"][0]  # old entry untouched
    assert patch(h, "EMP0001", "2026-07-06", status="LEAVE", punch_in=ms(2026, 7, 6, 9)).status_code == 422
    assert patch(h, "EMP0001", "2026-07-06", status="WFH").status_code == 422  # needs a punch_in
    r = patch(h, "EMP0001", "2026-07-06", status="WFH", punch_in=ms(2026, 7, 6, 9, 0), punch_out=ms(2026, 7, 6, 12, 0))
    assert r.status_code == 200 and r.json()["half_day"] is True and r.json()["work_hours"] == 3.0
    # validation
    assert patch(h, "EMP0001", "2026-07-06", punch_in=ms(2026, 7, 7, 9, 0)).status_code == 422  # leaves the date
    assert patch(h, "EMP0001", "2026-07-06", punch_out=ms(2026, 7, 6, 8, 0)).status_code == 422  # before punch_in
    assert patch(h, "EMP0001", "2026-07-06", punch_out=ms(2026, 7, 7, 9, 1)).status_code == 422  # > 24h
    assert patch(h, "EMP0001", "2026-07-06", punch_in=None).status_code == 422
    assert patch(h, "EMP0001", "2026-07-06", punch_in=1783312500).status_code == 422
    assert patch(h, "EMP0001", "2026-07-09", status="LEAVE").status_code == 404
    assert patch(h, "EMP9999", "2026-07-06", status="LEAVE").status_code == 404
    assert patch(h, "EMP0001", "2026-13-06", status="LEAVE").status_code == 422
    assert h.patch("/attendance/EMP0001/2026-07-06", json={"status": "LEAVE"}).status_code == 422  # reason/by required


def test_regularize_legacy_document(env):
    h = env.http
    add_emp(h, "EMP0002")
    env.db.attendance_logs.insert_one({
        "emp_code": "EMP0002", "date": "2026-07-14", "status": "PRESENT",
        "punch_in": datetime(2026, 7, 14, 4, 2, tzinfo=UTC), "punch_out": datetime(2026, 7, 14, 13, 1, tzinfo=UTC),
        "work_hours": 8.98, "late_minutes": 0, "overtime_minutes": 0,
    })
    rec = h.get("/attendance?emp_code=EMP0002").json()["items"][0]
    assert rec["history"] == [] and rec["half_day"] is False
    r = patch(h, "EMP0002", "2026-07-14", status="WFH")
    assert r.status_code == 200 and len(r.json()["history"]) == 1


def test_concurrent_regularize_keeps_every_history_entry(env):
    h = env.http
    add_emp(h, "EMP0001")
    make_day(h)
    times = [(9, m) for m in range(0, 10)]
    with ThreadPoolExecutor(10) as ex:
        res = [f.result() for f in [ex.submit(patch, h, "EMP0001", "2026-07-06", punch_in=ms(2026, 7, 6, *t)) for t in times]]
    codes = [r.status_code for r in res]
    assert set(codes) <= {200, 409}, codes
    final = h.get("/attendance?emp_code=EMP0001").json()["items"][0]
    assert len(final["history"]) == codes.count(200) >= 1


# --------------------------------------------------------------------------- #
# attendance list
# --------------------------------------------------------------------------- #
def test_attendance_list(env):
    h = env.http
    for c in ("EMP0001", "EMP0002", "EMP0003"):
        add_emp(h, c)
    for day in (6, 7, 8):
        for c in ("EMP0003", "EMP0001", "EMP0002"):
            h.post("/attendance/punch-in", json={"emp_code": c, "punched_at": ms(2026, 7, day, 9, 30), "status": "WFH" if c == "EMP0002" else "PRESENT"})
    items = h.get("/attendance?page_size=100").json()
    assert items["total"] == 9
    order = [(i["date"], i["emp_code"]) for i in items["items"]]
    assert order == sorted(order, key=lambda t: (-int(t[0].replace("-", "")), t[1]))  # date desc, emp_code asc
    p2 = h.get("/attendance?page=2&page_size=4").json()
    assert len(p2["items"]) == 4 and p2["total"] == 9
    f = h.get("/attendance?emp_code=EMP0001&date_from=2026-07-07&date_to=2026-07-08").json()
    assert f["total"] == 2 and [i["date"] for i in f["items"]] == ["2026-07-08", "2026-07-07"]
    assert h.get("/attendance?status=WFH").json()["total"] == 3
    assert h.get("/attendance?date_from=2026-07-09&date_to=2026-07-01").status_code == 422
    assert h.get("/attendance?date_from=2026-02-30").status_code == 422
    assert h.get("/attendance?status=BAD").status_code == 422
    assert h.get("/attendance?page=0").status_code == 422
    assert h.get("/attendance?page_size=101").status_code == 422


def test_health(env):
    assert env.http.get("/health").json() == {"status": "ok"}


# --------------------------------------------------------------------------- #
# analytics against an independent oracle (real MongoDB only)
# --------------------------------------------------------------------------- #
def hu(x, places):
    q = Decimal(1).scaleb(-places)
    return float((x if isinstance(x, Decimal) else Decimal(str(x))).quantize(q, rounding=ROUND_HALF_UP))


JULY = ("2026-07-01", "2026-07-31")


def build_dataset(db, seed=7):
    rnd = random.Random(seed)
    emps = []

    def emp(code, dept, joined, start="09:30", end="18:30"):
        emps.append({"emp_code": code, "name": "Name " + code, "email": code + "@x.com", "department": dept,
                     "shift_start": start, "shift_end": end, "joined_on": joined,
                     "created_at": datetime(2026, 1, 1, tzinfo=UTC)})

    for i in range(1, 9):
        emp(f"EMP{i:04d}", "Engineering", "2026-01-05")
    emp("EMP0009", "Engineering", "2026-07-15")  # joins mid-month
    emp("EMP0010", "Engineering", "2026-08-03")  # joins after July
    emp("EMP0011", "Engineering", "2026-01-05")  # never has a log
    for i in range(12, 17):
        emp(f"EMP{i:04d}", "Sales", "2026-02-01")
    emp("EMP0017", "Support", "2026-03-10", "22:00", "06:00")
    emp("EMP0018", "Support", "2026-03-10")
    emp("EMP0019", "Ops", "2026-09-01")  # department that did not exist yet in July
    db.employees.insert_many(emps)

    logs = []
    code_list = [e["emp_code"] for e in emps if e["emp_code"] not in ("EMP0011", "EMP0010", "EMP0019")]
    for code in code_list:
        joined = next(e["joined_on"] for e in emps if e["emp_code"] == code)
        for d in range(1, 32):
            day = date(2026, 7, d)
            if day.isoformat() < joined or rnd.random() < 0.15:  # a missing weekday is not an ABSENT record
                continue
            roll = rnd.random()
            doc = {"emp_code": code, "date": day.isoformat(), "punch_in": None, "punch_out": None, "work_hours": None,
                   "late_minutes": 0, "overtime_minutes": 0, "half_day": False, "history": []}
            if roll < 0.08:
                doc["status"] = "ABSENT"
            elif roll < 0.16:
                doc["status"] = "LEAVE"
            else:
                doc["status"] = rnd.choice(["PRESENT", "PRESENT", "WFH", "ON_DUTY"])
                doc["punch_in"] = datetime(2026, 7, d, 4, 0, tzinfo=UTC)
                if rnd.random() < 0.9:
                    wh = round(rnd.choice([3.5, 4.0, 4.49, 8.0, 8.5, 9.01, 9.25, 10.0]) + rnd.choice([0, 0.01, 0.02]), 2)
                    doc["work_hours"] = wh
                    doc["punch_out"] = doc["punch_in"] + timedelta(hours=wh)
                    doc["half_day"] = wh < 4.5
                    doc["overtime_minutes"] = rnd.choice([0, 0, 30, 45, 60])
                doc["late_minutes"] = rnd.choice([0, 0, 0, 11, 20, 20, 35, 60])
            if rnd.random() < 0.1:  # legacy shape
                doc.pop("half_day"), doc.pop("history")
            logs.append(doc)
    # identical totals on purpose -> a three-way tie (these employees get exactly one late record)
    tie_codes = ("EMP0012", "EMP0013", "EMP0014")
    logs = [d for d in logs if d["emp_code"] not in tie_codes]
    for code, mins in ((c, 50) for c in tie_codes):
        logs.append({"emp_code": code, "date": "2026-07-20", "status": "PRESENT", "punch_in": datetime(2026, 7, 20, 4, 0, tzinfo=UTC),
                     "punch_out": None, "work_hours": None, "late_minutes": mins, "overtime_minutes": 0, "half_day": False, "history": []})
    logs.append({"emp_code": "GHOST01", "date": "2026-07-20", "status": "PRESENT", "punch_in": datetime(2026, 7, 20, 4, 0, tzinfo=UTC),
                 "punch_out": None, "work_hours": None, "late_minutes": 500, "overtime_minutes": 0, "half_day": False, "history": []})
    # records in neighbouring months must be ignored by July analytics
    logs.append({"emp_code": "EMP0001", "date": "2026-06-30", "status": "PRESENT", "punch_in": datetime(2026, 6, 30, 4, 0, tzinfo=UTC),
                 "punch_out": None, "work_hours": None, "late_minutes": 90, "overtime_minutes": 0, "half_day": False, "history": []})
    db.attendance_logs.insert_many(logs)
    return emps


def present(doc):
    return doc["status"] in ("PRESENT", "WFH", "ON_DUTY")


def credit(doc):
    return Decimal("0.5") if doc.get("half_day") else Decimal(1)


def weekday(day_str):
    return date.fromisoformat(day_str).weekday() < 5


def oracle_working_days(joined, first, last):
    start = max(first, joined)
    if start > last:
        return 0
    d, n = date.fromisoformat(start), 0
    while d.isoformat() <= last:
        n += d.weekday() < 5
        d += timedelta(days=1)
    return n


@needs_mongo
def test_employee_monthly_vs_oracle(env):
    emps = build_dataset(env.db)
    first, last = JULY
    for e in emps:
        logs = list(env.db.attendance_logs.find({"emp_code": e["emp_code"], "date": {"$gte": first, "$lte": last}}))
        wd = oracle_working_days(e["joined_on"], first, last)
        pdays = sum((credit(d) for d in logs if present(d) and weekday(d["date"])), Decimal(0))
        exp = {
            "emp_code": e["emp_code"], "month": "2026-07", "working_days": wd, "present_days": hu(pdays, 2),
            "leave_days": sum(d["status"] == "LEAVE" for d in logs),
            "late_count": sum(d["late_minutes"] > 0 for d in logs),
            "total_late_minutes": sum(d["late_minutes"] for d in logs),
            "total_overtime_minutes": sum(d["overtime_minutes"] for d in logs),
            "attendance_pct": None if wd == 0 else hu(pdays * 100 / wd, 2),
        }
        r = env.http.get(f"/analytics/employees/{e['emp_code']}/monthly?month=2026-07")
        assert r.status_code == 200, r.text
        assert r.json() == exp, e["emp_code"]
    assert env.http.get("/analytics/employees/NOPE/monthly?month=2026-07").status_code == 404
    assert env.http.get("/analytics/employees/EMP0001/monthly?month=2026-13").status_code == 422
    assert env.http.get("/analytics/employees/EMP0001/monthly").status_code == 422


@needs_mongo
def test_department_summary_vs_oracle(env):
    emps = build_dataset(env.db)
    first, last = JULY
    expected = {}
    for e in emps:
        if e["joined_on"] > last:
            continue
        g = expected.setdefault(e["department"], dict(headcount=0, pd=Decimal(0), wh=[], lc=0, ltm=0, lv=0, od=0))
        g["headcount"] += 1
        for d in env.db.attendance_logs.find({"emp_code": e["emp_code"], "date": {"$gte": first, "$lte": last}}):
            if present(d) and weekday(d["date"]):
                g["pd"] += credit(d)
            if present(d) and d["work_hours"] is not None:
                g["wh"].append(Decimal(str(d["work_hours"])))
            g["lc"] += d["late_minutes"] > 0
            g["ltm"] += d["late_minutes"]
            g["lv"] += d["status"] == "LEAVE"
            g["od"] += d["status"] == "ON_DUTY"
    items = [
        {"department": k, "headcount": g["headcount"], "present_days": hu(g["pd"], 2),
         "avg_work_hours": hu(sum(g["wh"]) / len(g["wh"]), 2) if g["wh"] else None,
         "late_count": g["lc"], "total_late_minutes": g["ltm"], "leave_count": g["lv"], "on_duty_count": g["od"]}
        for k, g in sorted(expected.items())
    ]
    r = env.http.get("/analytics/departments/summary?month=2026-07")
    assert r.status_code == 200, r.text
    assert r.json() == {"month": "2026-07", "items": items}
    assert "Ops" not in [i["department"] for i in r.json()["items"]]  # headcount 0 -> omitted
    eng = next(i for i in items if i["department"] == "Engineering")
    assert eng["headcount"] == 10  # EMP0011 has no logs but counts; EMP0010 joins in August and does not
    one = env.http.get("/analytics/departments/summary?month=2026-07&department=Sales").json()
    assert one["items"] == [i for i in items if i["department"] == "Sales"]
    assert env.http.get("/analytics/departments/summary?month=2026-07&department=Nope").json()["items"] == []
    assert env.http.get("/analytics/departments/summary").status_code == 422


@needs_mongo
def test_late_leaderboard_vs_oracle(env):
    emps = build_dataset(env.db)
    first, last = JULY
    by_code = {e["emp_code"]: e for e in emps}

    def board(dept=None):
        tot = {}
        for d in env.db.attendance_logs.find({"date": {"$gte": first, "$lte": last}}):
            if d["emp_code"] not in by_code or d["late_minutes"] <= 0:
                continue
            if dept and by_code[d["emp_code"]]["department"] != dept:
                continue
            t = tot.setdefault(d["emp_code"], [0, 0])
            t[0] += d["late_minutes"]
            t[1] += 1
        rows = sorted(tot.items(), key=lambda kv: (-kv[1][0], kv[0]))
        out, prev, rank = [], None, 0
        for pos, (code, (m, c)) in enumerate(rows, start=1):
            if m != prev:
                rank, prev = pos, m
            out.append({"rank": rank, "emp_code": code, "name": by_code[code]["name"], "department": by_code[code]["department"],
                        "total_late_minutes": m, "late_count": c})
        return out

    for dept in (None, "Engineering", "Sales", "Ops"):
        full = board(dept)
        for limit in (1, 2, 3, 5, 10, 50):
            url = f"/analytics/leaderboard/late?month=2026-07&limit={limit}" + (f"&department={dept}" if dept else "")
            r = env.http.get(url)
            assert r.status_code == 200, r.text
            assert r.json() == {"month": "2026-07", "items": [x for x in full if x["rank"] <= limit]}, (dept, limit)
    ties = [x for x in board() if x["total_late_minutes"] == 50]
    assert len(ties) >= 3 and len({x["rank"] for x in ties}) == 1  # the tie group shares one rank
    assert "GHOST01" not in [x["emp_code"] for x in board()]
    assert env.http.get("/analytics/leaderboard/late?month=2026-07&limit=0").status_code == 422
    assert env.http.get("/analytics/leaderboard/late?month=2026-07&limit=51").status_code == 422


def oracle_trend(db, dept, frm, to):
    emps = list(db.employees.find({"department": dept}))
    codes = {e["emp_code"] for e in emps}
    day, rows = date.fromisoformat(frm), []
    while day <= date.fromisoformat(to):
        s = day.isoformat()
        recs = [d for d in db.attendance_logs.find({"date": s}) if d["emp_code"] in codes]
        hc = sum(e["joined_on"] <= s for e in emps)
        wd = day.weekday() < 5
        pc = sum((credit(d) for d in recs if present(d)), Decimal(0))
        rate = hu(pc / hc, 4) if wd and hc > 0 else None
        rows.append({"date": s, "is_working_day": wd, "headcount": hc, "present_count": float(pc),
                     "late_count": sum(d["late_minutes"] > 0 for d in recs), "attendance_rate": rate})
        day += timedelta(days=1)
    for i, row in enumerate(rows):
        window = [r["attendance_rate"] for r in rows[max(0, i - 6): i + 1] if r["attendance_rate"] is not None]
        row["moving_avg_7d"] = hu(sum(Decimal(str(x)) for x in window) / len(window), 4) if window else None
    return rows


@needs_mongo
def test_department_trend_vs_oracle(env):
    build_dataset(env.db)
    for dept, frm, to in (
        ("Engineering", "2026-06-25", "2026-08-10"),  # crosses month ends, covers the mid-month joiner
        ("Sales", "2026-07-01", "2026-07-31"),
        ("Support", "2026-07-04", "2026-07-12"),
        ("Ops", "2026-08-25", "2026-09-05"),  # nobody has logs; headcount appears when people join
        ("Engineering", "2026-12-01", "2026-12-01"),  # a single day with no records
    ):
        r = env.http.get(f"/analytics/departments/{dept}/trend", params={"from": frm, "to": to})
        assert r.status_code == 200, r.text
        got = r.json()
        assert got["department"] == dept
        assert got["items"] == oracle_trend(env.db, dept, frm, to), (dept, frm, to)
    one_day = env.http.get("/analytics/departments/Engineering/trend", params={"from": "2026-12-01", "to": "2026-12-01"}).json()["items"]
    assert one_day[0]["attendance_rate"] == 0.0 and one_day[0]["moving_avg_7d"] == 0.0
    sat = env.http.get("/analytics/departments/Engineering/trend", params={"from": "2026-07-04", "to": "2026-07-04"}).json()["items"][0]
    assert sat["is_working_day"] is False and sat["attendance_rate"] is None and sat["moving_avg_7d"] is None
    assert env.http.get("/analytics/departments/Nope/trend", params={"from": "2026-07-01", "to": "2026-07-02"}).status_code == 404
    assert env.http.get("/analytics/departments/Sales/trend", params={"from": "2026-07-02", "to": "2026-07-01"}).status_code == 422
    assert env.http.get("/analytics/departments/Sales/trend", params={"from": "2026-07-01", "to": "2026-09-30"}).status_code == 200  # 92 days
    assert env.http.get("/analytics/departments/Sales/trend", params={"from": "2026-07-01", "to": "2026-10-01"}).status_code == 422  # 93 days
    assert env.http.get("/analytics/departments/Sales/trend", params={"from": "2026-07-01"}).status_code == 422


@needs_mongo
def test_explain_uses_indexes(env):
    build_dataset(env.db)
    calls = {
        "attendance_list": {},
        "attendance_list ": {"emp_code": "EMP0001", "date_from": "2026-07-01", "date_to": "2026-07-31"},
        "attendance_list  ": {"status": "WFH"},
        "employee_monthly": {"emp_code": "EMP0001", "month": "2026-07"},
        "department_summary": {"month": "2026-07"},
        "department_summary ": {"month": "2026-07", "department": "Sales"},
        "late_leaderboard": {"month": "2026-07", "limit": 5},
        "department_trend": {"department": "Engineering", "from": "2026-07-01", "to": "2026-07-31"},
    }
    import json

    for name, params in calls.items():
        r = env.http.get(f"/admin/explain/{name.strip()}", params=params)
        assert r.status_code == 200, (name, r.text)
        text = json.dumps(r.json()["explain"])
        assert "IXSCAN" in text, name
        assert "COLLSCAN" not in text, name
    assert env.http.get("/admin/explain/employee_monthly", params={"emp_code": "EMP0001"}).status_code == 422
    assert env.http.get("/admin/explain/department_trend", params={"department": "Sales"}).status_code == 422
    assert env.http.get("/admin/explain/nonsense").status_code == 422
