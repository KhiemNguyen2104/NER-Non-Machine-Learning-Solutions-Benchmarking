# A non-machine learning based named entity recognition (NER) system

## Dependencies Installation

Create a Python environment:

- `python -m venv .venv`

Activate the Python environment:

- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

- `python -m pip install -r requirements.txt`

## Run the program

After setup you can run the program in the following ways:

- `python main.py setup`              # one-time data download (~5 min)
- `python main.py demo`               # run both pipelines on 8 example sentences
- `python main.py run-chunker`        # interactive session
- `python main.py run-cfg`            # interactive session  
- `python main.py benchmark`          # all 4 experiments + plots
