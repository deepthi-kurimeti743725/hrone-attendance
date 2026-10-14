# DECISIONS.md

1. **Indexes.** `employees`: unique `emp_code` (duplicate guarantee, joins), `(department, joined_on)` and `joined_on` (headcount, trend). `attendance_logs`: unique `(emp_code, date)` (one record per day, per-employee queries); `(date desc, emp_code)` (the list sort and month-range analytics); `(status, date desc, emp_code)` (status filter); `(emp_code, punch_in desc)` (finding the record to punch out). I rejected an index on `late_minutes`: the date index narrows to a month first, so it would add write cost for little gain.

2. **Punch-in race.** Both requests build a record and call `insert_one`. The unique `(emp_code, date)` index lets exactly one succeed; the other gets `DuplicateKeyError`, which I turn into 409. There is no find-then-insert.

3. **Ties.** Ranking uses `$rank` on total late minutes, so tied employees share a rank and the next is skipped. `limit` is applied after ranking, so a tie at the cutoff returns all tied people, more than `limit` rows, ordered by `emp_code`.

4. **Headcount.** The summary starts from `employees` (joined on or before month end), then looks up each person's logs. No logs means an empty lookup, but the employee row is still counted.

5. **100x data.** I would stop aggregating raw logs per request and keep a daily roll-up per department (updated at punch time or by a job), so analytics read a few rows per day.
