# Data setup

Banking77 is a static intent dataset. The course vignette treats new chatbot messages as rows that keep arriving in an OLTP log, often with `label` still null. These scripts copy the local Banking77 files, clean them, and simulate that log on disk. S3 is not wired up yet.

Fraud flags in the stream are simulated. They are not labels from the bank or from Banking77.

## Run

From the repo root, with Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/data/prepare_banking77.py
python src/data/simulate_stream.py --profile dev
```

`prepare_banking77.py` only reads `../mid-term/`. It copies those files and writes under `data/`.

`simulate_stream.py` reads `config/simulation.yaml`. The default profile is `dev`: 10,000 users and 500,000 messages. Pass `--profile full` to generate 1,000,000 users and 50,000,000 messages. `full` does not start just because `active_profile` is changed to `full`; the flag is required. Both profiles write Parquet in chunks, partitioned by date, so the 50 million rows are not held in one frame.

A rerun deletes `data/simulated/messages/` and rewrites the simulated tables. The seed is `simulation.yaml` `seed` (7, the same value as the midterm practice sample). User-level draws use `SeedSequence([seed, user_id, stream])`.

Paths come from `config/storage.yaml`. `backend: local` writes `data/`. `backend: s3` builds an `s3a://` URI and refuses to write until `s3_bucket` and `s3_prefix` are set. `S3_BUCKET` and `S3_PREFIX` override the file. Do not put AWS keys in the yaml. If you add keys later, put them in `.env`, which is gitignored.

```bash
./scripts/upload_to_s3.sh
```

The script exits with a message and uploads nothing while the bucket is `TODO`.

## Folder layout

```text
config/intent_taxonomy.yaml    journey stage, fraud type, and risk for all 77 intents
config/simulation.yaml         seed, profile sizes, cities, fraud-pattern rules
config/storage.yaml            local root, or the S3 placeholder
config/prepare.yaml            input paths and dim_intent defaults
src/data/prepare_banking77.py
src/data/simulate_stream.py
src/data/storage.py            local path vs s3a://
scripts/upload_to_s3.sh
data/raw/                      copy of train, test, and intents.csv
data/processed/                cleaned sentences and dim_intent
data/simulated/messages/       date=YYYY-MM-DD/part-#####.parquet
data/simulated/users.parquet
data/simulated/daily_report.parquet
```

`data/raw/`, `data/processed/`, `data/simulated/`, and `.env` are gitignored.

## Schemas

### `data/raw/train.parquet` and `data/raw/test.parquet`

The unmodified Banking77 splits. 10,003 train rows and 3,080 test rows.

| column | type | meaning |
| --- | --- | --- |
| `text` | string | customer message |
| `label` | int | intent id 0–76 |

### `data/raw/intents.csv`

| column | type | meaning |
| --- | --- | --- |
| `label` | int | intent id |
| `intent` | string | intent name, exactly as published |

### `data/processed/banking77_clean.parquet`

Train and test stacked. 13,083 rows.

| column | type | meaning |
| --- | --- | --- |
| `text` | string | whitespace collapsed and trimmed |
| `label` | int | intent id |
| `intent` | string | name from `intents.csv` |
| `split` | string | `train` or `test` |
| `text_length` | int | characters after cleaning |

There is no `received` column here. The original files have no event time.

### `data/processed/dim_intent.parquet`

One row per intent (77). Same core columns as the midterm practice `dim_intent`, plus the taxonomy. `fraud_relevant` is true exactly when `fraud_type` is set (22 intents). Every current row uses `intent_set_version` 1, `valid_from` 2020-01-01, and `valid_to` 9999-12-31. A later intent would be a new row with its own version and window, not a new column.

| column | type | meaning |
| --- | --- | --- |
| `intent_id` | int | 0–76 |
| `intent` | string | intent name |
| `fraud_relevant` | bool | intent is in a fraud type |
| `valid_from` | timestamp | SCD2 start |
| `valid_to` | timestamp | SCD2 end; 9999-12-31 means still current |
| `intent_set_version` | int | catalog version, 1 for this file |
| `journey_stage` | string | session theme, such as `card_setup` or `transfers` |
| `fraud_type` | string or null | taxonomy class, or null |
| `risk_level` | string | `low`, `medium`, or `high` |

Fraud types and risk:

| fraud_type | risk | intents |
| --- | --- | --- |
| `account_takeover` | high | `change_pin`, `passcode_forgotten`, `pin_blocked`, `edit_personal_details`, `lost_or_stolen_phone` |
| `card_compromise` | high | `compromised_card`, `lost_or_stolen_card`, `card_payment_not_recognised`, `cash_withdrawal_not_recognised` |
| `app_fraud_money_mule` | medium | `beneficiary_not_allowed`, `transfer_not_received_by_recipient`, `receiving_money`, `cancel_transfer` |
| `identity_kyc_bypass` | high | `verify_my_identity`, `unable_to_verify_identity`, `why_verify_identity` |
| `virtual_card_abuse` | medium | `get_disposable_virtual_card`, `disposable_card_limits`, `virtual_card_not_working` |
| `refund_chargeback_abuse` | medium | `request_refund`, `Refund_not_showing_up`, `reverted_card_payment?` |

`prepare_banking77.py` stops if any of these names is missing from `intents.csv`, or if the roster and the per-intent fields disagree. Names are copied exactly, including `Refund_not_showing_up` and `reverted_card_payment?`.

### `data/simulated/users.parquet`

| column | type | meaning |
| --- | --- | --- |
| `user_id` | int | 1…n |
| `home_city` | string | city the user normally writes from |
| `home_lat`, `home_lon` | double | city center |
| `persona` | string | `normal`, or one fraud pattern name |
| `n_messages` | int | rows for this user |
| `n_fraud_messages` | int | rows with `is_fraud`; 0 for normal users |

### `data/simulated/messages/`

Hive layout: `date=YYYY-MM-DD/part-#####.parquet`. `date` is the partition key and is not repeated inside the file. A chunk is flushed after a whole user, so part files are about `chunk_size` rows, not exactly that size. `message_id` is assigned in `user_id` order, and within a user in time order. It is not global time order.

The OLTP columns match the midterm `message` table: `message_id`, `received`, `text`, `label`, `label_source`, `intent_set_version`. The stream adds the columns below that.

| column | type | meaning |
| --- | --- | --- |
| `message_id` | int | stream key, starting at 1 |
| `received` | string | `YYYY-MM-DD_HH:MM:SS`, same format as the vignette |
| `received_ts` | timestamp | parsed event time |
| `text` | string | a real Banking77 sentence for the sampled intent |
| `label` | int or null | intent id, kept on about 30% of rows |
| `label_source` | string or null | `manual` when `label` is kept, otherwise null |
| `intent_set_version` | int | 1 |
| `user_id` | int | |
| `session_id` | int | one burst of messages |
| `city`, `lat`, `lon` | string, double, double | usually the home city; an ordinary trip or a `location_anomaly` burst can be elsewhere |
| `is_fraud` | bool | true only on rows the pattern rules flag |
| `fraud_pattern` | string or null | pattern name, or null when `is_fraud` is false |

`label` is an intent id, not a fraud flag. Null `label` means nobody has labeled that message yet. `is_fraud` is generator ground truth and is filled on every row.

### `data/simulated/daily_report.parquet`

One row per calendar day, from the written Parquet, not from the OLTP source.

| column | type | meaning |
| --- | --- | --- |
| `day` | timestamp | calendar day from `received_ts` (midnight) |
| `messages` | int | rows that day |
| `labeled_messages` | int | rows with a non-null `label` |
| `manual_coverage` | double | share with `label_source = manual`, rounded to 3 decimals |
| `fraud_rate` | double | share with `is_fraud`, rounded to 3 decimals |

## Fraud rules

Most ordinary sessions are at home, between 08:00 and 20:00, and stay inside one journey stage. Ordinary sessions are also the background of fraud users. These shares, set under `ordinary_sessions` in `config/simulation.yaml`, keep a single field from lining up with `is_fraud`:

- `normal_night_share` (0.03): that share of ordinary sessions falls between 00:00 and 05:00, the same clock window as `night_activity`. Each of those sessions is still only 2–5 messages over at most 25 minutes. The fraud pattern remains one cluster of 6–12 messages packed into 1–3 hours. Hour alone does not separate them.
- `normal_evening_share` (0.08): that share falls between 20:00 and 24:00. The rest stay in the daytime window.
- `normal_away_session_share` (0.03): that share is a trip to another city at least `min_away_km` away. A city change is not by itself `location_anomaly`.
- `sensitive_intent_share` (0.05): that share uses exactly one intent from `sensitive_intents` (`declined_card_payment`, `receiving_money`, `change_pin`, `passcode_forgotten`, `edit_personal_details`, and the other names in the file). The session does not run the takeover chain and does not grow into a card-testing burst.

About 8% of users are assigned one pattern (`fraud_user_share`). Only the pattern burst is `is_fraud`. Counts, windows, and intent lists are all in `config/simulation.yaml`.

`fraud_pattern` is not `dim_intent.fraud_type`. Card testing, for example, is mostly `declined_card_payment`, which also shows up in ordinary sessions. The flag comes from the burst, not from the intent name.

1. `account_takeover`. Account takeover: reset the PIN or passcode, change personal details, then move money. Banking77 has no "add beneficiary" intent, so the last step uses `beneficiary_not_allowed` and other transfer intents. The three steps stay in that order inside a 10–40 minute window.
2. `money_mule`. Money mule: several `receiving_money` messages, then a fast burst of outgoing transfer intents.
3. `card_testing`. Card testing: a short burst of `declined_card_payment` mixed with limit questions (`top_up_limits`, `disposable_card_limits`).
4. `location_anomaly`. Impossible travel: one later session, after the user's earlier history, is in another city at least `min_away_km` (800) away. An ordinary trip uses the same distance rule but is not flagged.
5. `night_activity`. One tight cluster between 00:00 and 05:00, longer than an ordinary night session. Other late-night messages, including this user's own daytime background and other users' short night sessions, are not flagged.

## Limitations

- `is_fraud` is assigned by the rules above. No investigator labeled these streams.
- The sentences are real Banking77 utterances, but user, session, city, time, and the fraud flag are simulated.
- A sentence is only a loose match for its intent. A `declined_transfer` row can still read like a declined card payment.
- A few fraud-typed intents also appear in ordinary sessions (`sensitive_intents`). The fraud label is the pattern, not the intent. Intents that are neither ordinary nor on that list still never occur.
- Keeping `label` on about 30% of rows is the manual-label process from the midterm vignette. It is not a sample of confirmed fraud.
- The dev stream is one month (2026-10-01 through 2026-10-31) and fits on a laptop. It is not a production volume.
- S3 upload and Spark reads are not implemented. `storage.py` only switches the URI between a local path and `s3a://`.
