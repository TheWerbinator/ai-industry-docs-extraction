# ⚙️ Industrial Spec-to-JSON Engine

## 🛠️ The Mission
In industrial automation, data is often trapped in legacy PDF catalogs or technical datasheets. This project is a **Deterministic Extraction Pipeline** designed to ingest unstructured industrial PDFs and transform them into validated, machine-readable JSON schemas.

**Key Technical Highlight:** Unlike a standard chatbot, this system uses **Pydantic-based Output Parsing** to ensure that the AI follows a strict "Technical Contract." If the AI cannot find a value or tries to hallucinate a format, the system flags it during validation.

---

## 🚀 Getting Started

### 1. System Dependencies (CRITICAL)
This engine uses `unstructured.io` to partition complex PDF schemas. You **must** install the following system-level tools for the PDF partitioning to work:

* **macOS (via Homebrew):**
    ```bash
    brew install poppler tesseract
    ```
* **Windows:**
    1.  Download and install [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/).
    2.  Download and install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
    3.  Add the `/bin` folders of both to your **System Environment PATH**.

### 2. Python Environment Setup
```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# (Windows)
.\venv\Scripts\activate
# (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Program

After all the above is complete, run the project with:

```bash
streamlit run app.py
```

You will be prompted to input your OpenAI product key and the associated catalog PDF.

**Once those are in and processed, give it a prompt!**

The output is rigid - the code is telling it only to output JSON for a motor's Model Number, Voltage, Horsepower, RPM, Size, and Enclosure.

As a proof of concept, this whole project is essentially showing that we can extract unstructured data and transform it into highly structured, readable JSON.