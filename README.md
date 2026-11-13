# DSAN6000-Project

Scalable fraud detection in banking chatbot conversations.

Banking77 supplies the labeled customer sentences. This repo turns that static file into a local OLTP-style message stream: most new messages have no intent label, a minority are manually labeled, and a small share follow simulated fraud patterns. Later work can point the same paths at S3 and Spark. The bucket is still a placeholder.

## Data quick start

From this directory, with Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/data/prepare_banking77.py
python src/data/simulate_stream.py --profile dev
```

`dev` builds 10,000 users and 500,000 messages. `python src/data/simulate_stream.py --profile full` builds the 1,000,000-user / 50,000,000-message stream in chunks. Generated files under `data/` are gitignored. Details, column schemas, and the fraud rules are in [docs/data_setup.md](docs/data_setup.md).
