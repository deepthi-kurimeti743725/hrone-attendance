# Employee Attendance & Analytics API

FastAPI + MongoDB (6.0+). All application code is in `app/main.py`.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MONGO_URI="mongodb://localhost:27017" MONGO_DB="attendance_db"   # or a local .env
uvicorn app.main:app --port 8000
```
Indexes are created at startup (idempotent). Local MongoDB: `docker run -p 27017:27017 mongo:7`.

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q                                       # rules + CRUD (in-memory mock)
MONGO_URI=mongodb://localhost:27017 pytest -q   # everything, incl. analytics vs a plain-Python oracle and explain
```
Real-MongoDB tests use a throw-away database and drop it afterwards.

## Status
All endpoints in `openapi.yaml` are implemented. `REVIEW.md` and `DECISIONS.md` are filled in.

## Things to know
- `moving_avg_7d` is the mean of the displayed (4-decimal, half-up) `attendance_rate` values in the window, rounded half-up again to 4 decimals.
- Half-up rounding is done with exact integer / `Decimal` maths, never `$round` (which rounds half to even) or float `round()`.
- `PATCH` answers 409 if the record changed between reading and writing (optimistic check on history length, times and status), so no audit entry is lost.
- Trend range limit is 92 days inclusive.
