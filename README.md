# Automatic Essay Scoring — Streamlit App

This application scores an essay with a trained machine-learning model and displays:

- a predicted score and confidence estimate;
- word and sentence statistics;
- vocabulary, readability, grammar, spelling, and transition indicators;
- local feature importance;
- strengths and suggested areas for improvement.

The trained model files are included in the project, so you do not need to train a model or download a dataset before running the app.

## Requirements

- Python 3.10 or newer (Python 3.11 is recommended)
- `pip`
- About 1 GB of free disk space for the Python environment and dependencies
- An internet connection during the initial dependency installation

Java is optional. If it is available, `language-tool-python` can provide enhanced grammar checks. Without Java, the application automatically uses its built-in grammar rules.

## Project files required at runtime

Keep this structure intact:

```text
Automatic-Essay-Scoring/
├── app.py
├── config.yaml
├── inference.py
├── requirements.txt
├── checkpoints/
│   └── best_model/
│       ├── baseline_model.joblib
│       ├── feature_scaler.joblib
│       ├── label_metadata.json
│       └── text_vectorizer.joblib
├── feature_engineering/
├── preprocessing/
└── utils/
```

Do not rename or remove files under `checkpoints/best_model`; they are required for predictions.

## Installation

Open a terminal in the project directory.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, run this once in the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the application

With the virtual environment activated and the terminal still in the project directory, run:

```bash
streamlit run app.py
```

Streamlit should open the application automatically. If it does not, visit:

```text
http://localhost:8501
```

To stop the application, return to the terminal and press `Ctrl+C`.

## Using the app

1. Optionally enter the prompt name, assignment instructions, and source text in the sidebar.
2. Paste the student's essay into the **Student essay** box.
3. Select **Score essay**.
4. Review the predicted score, confidence, writing profile, feature importance, and feedback.

The grammar and spelling indicators are automated estimates. They should support human review, not replace it. Flesch Reading Ease normally ranges from 0 (very difficult) to 100 (very easy), although exceptionally dense writing can produce a value below 0.

## Troubleshooting

### `python` or `streamlit` is not recognized

Confirm that Python is installed and that the virtual environment is activated. You can also launch Streamlit through Python:

```bash
python -m streamlit run app.py
```

### Model artifacts not found

Confirm that all four files are present under `checkpoints/best_model` and that you are running the command from the project root—the directory containing `app.py`.

### Dependency installation fails

Upgrade the installation tools and retry:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If the failure refers to an unsupported Python version, install Python 3.11, recreate `.venv`, and reinstall the requirements.

### Enhanced language tools are unavailable

The app has built-in fallbacks and will continue to run. For enhanced grammar checking, install a current Java runtime and restart Streamlit. The optional spaCy language model can be installed with:

```bash
python -m spacy download en_core_web_sm
```

The app also works without that optional model by using its rule-based language pipeline.

## Deactivate the environment

When finished, stop Streamlit with `Ctrl+C`, then run:

```bash
deactivate
```
