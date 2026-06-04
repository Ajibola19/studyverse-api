# PDFQuizzer — AI-Powered MCQ Generator from PDFs

This system ingests a PDF document, performs intelligent text extraction and summarization, and automatically generates structured Multiple-Choice Questions (MCQs) using LLMs. It produces clean, well-formatted questions — each with four options and a revealable correct answer — through a simple web interface. Designed for students, teachers, trainers, and learning platforms who want fast MCQ creation without manual typing.

---

## Key Features

### AI-Driven Question Engine

**PDF Text Extraction :** Uses PyPDF2 to safely extract readable text from uploaded PDF files, page-by-page, with Unicode handling and fallback support.

**Context Summarization :** Summarizes long text using LangChain’s map-reduce pipeline to create focused context for reliable MCQ generation.

**Topic-Aware Questioning :** Optional topic filter lets users generate questions specifically around a selected topic — such as *Photosynthesis*, *Neural Networks*, or *World War II*.

**MCQ Generation :** Uses Groq-powered LLaMA-3.3-70B to create well-structured questions with:

* 4 answer options (a, b, c, d)
* Exactly one correct answer
* Clean formatting for UI parsing
---

### Technology Stack

**Backend :**
Python, Flask, LangChain, Groq LLM API, PyPDF2, python-dotenv

**Frontend :**
HTML, CSS (custom), Jinja2 templating

**LLM & AI Layer :**
LLaMA-3.3-70B-Versatile via Groq API (fast inference)

**Deployment :**
Cloud-ready — tested on Render
---

## Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/BhaveshBhakta/PDFQuizzer-A-Smart-PDF-MCQ-Generator.git
cd PDFQuizzer-A-Smart-PDF-MCQ-Generator
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
```

Run the app:

```bash
python app.py
```

Open the UI:

```
http://localhost:5000
```
## High-Level Architecture

User (PDF Upload + Settings)
   ↓
Flask Web App
   ↓

┌────────────── Intelligent Pipeline ───────────────┐
│ PDF Upload & Text Extraction                      │
│ Text Chunking & Summarization                     │
│ Topic-Sensitive MCQ Generation via Groq LLM       │
│ Output Formatting & UI Rendering                  │
└───────────────────────────────────────────────────┘
   ↓
Interactive MCQ Viewer (Reveal Answers UI)

---

## Roadmap & Future Enhancements

* Export MCQs to Google Forms automatically
* Download as PDF / CSV / JSON
* Quiz-taking mode with scoring
* Difficulty level selection
* Multi-chapter topic detection
* Save past sessions for users
* Classroom dashboard integration
* Multi-language PDF support
* LMS plugin support (Moodle / Canvas)
