# 📄 AI Resume Parser & Candidate Ranking System

An AI-powered recruitment screening application that extracts structured information from resumes, analyses job descriptions, evaluates candidate skill matches, and ranks applicants based on their suitability for a role.

**Live Demo:** [AI Resume Parser](https://akshatsinghrawat-ai-resume-parser-app-oyehwc.streamlit.app)

**GitHub Repository:** [ai-resume-parser](https://github.com/akshatsinghrawat/ai-resume-parser)

---

## 🚀 Project Overview

Recruiters often receive a large number of resumes for a single job opening. Manually reviewing each resume, identifying relevant skills, and comparing candidates against job requirements can be time-consuming.

This project aims to simplify the initial screening process by using Large Language Models (LLMs) to extract structured information from job descriptions and resumes.

The application compares candidate skills against the job requirements, calculates match scores, ranks candidates, and provides individual candidate reports with supporting evidence.

The system is designed to assist recruiters with initial screening. Its scores represent skill overlap and should not be treated as final hiring decisions.

## ✨ Key Features

- **Job Description Parsing:** Extracts job roles, required skills, preferred skills, education requirements, and responsibilities from uploaded job descriptions.

- **Resume Parsing:** Extracts candidate details, contact information, technical skills, education, work experience, internships, projects, and certifications.

- **Structured Data Extraction:** Uses Groq-powered LLM responses and Pydantic models to convert unstructured documents into structured data.

- **Skill Matching:** Compares candidate skills against required and preferred job skills using skill normalization and predefined aliases.

- **Candidate Scoring:** Calculates individual required-skill and preferred-skill scores and combines them into an overall match score.

- **Candidate Ranking:** Ranks multiple candidates according to their overall match scores.

- **Experience Categorization:** Separates professional employment, internships, leadership positions, and other experience.

- **Project Skill Extraction:** Identifies technical skills explicitly mentioned in candidate project descriptions.

- **Individual Candidate Reports:** Displays matching skills, missing skills, score breakdowns, experience details, and evidence supporting matched skills.

- **CSV Export:** Allows recruiters to download candidate evaluation results for further analysis.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM Integration | Groq API |
| Language Model | openai/gpt-oss-120b |
| Data Validation | Pydantic |
| Data Processing | Pandas |
| PDF Processing | PyPDF |
| DOCX Processing | python-docx |
| Environment Management | python-dotenv |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

## ⚙️ How It Works

The application follows a structured processing pipeline:

1. **Upload Documents:** The recruiter uploads a job description and multiple candidate resumes.

2. **Text Extraction:** Text is extracted from supported PDF, DOCX, and TXT job descriptions and PDF/DOCX resumes.

3. **Job Description Analysis:** The Groq API processes the job description and identifies required and preferred technical skills.

4. **Resume Analysis:** Each resume is processed using the LLM to extract candidate information, skills, experience, education, projects, and certifications.

5. **Data Validation:** Pydantic validates the extracted information against predefined data models.

6. **Skill Matching:** Candidate skills are normalized and compared against the job requirements.

7. **Score Calculation:** Required and preferred skill scores are calculated and combined using a weighted scoring formula.

8. **Candidate Ranking:** Candidates are sorted according to their overall match scores.

9. **Report Generation:** Individual candidate reports and downloadable CSV results are generated.

## 📊 Candidate Scoring Methodology

The application uses a weighted skill-matching approach.

**Scoring Formula:**

Overall Match Score = (Required Skills Score × 0.80) + (Preferred Skills Score × 0.20)

Where:

- Required Skills Score = (Number of Matching Required Skills / Total Required Skills) × 100
- Preferred Skills Score = (Number of Matching Preferred Skills / Total Preferred Skills) × 100

If only one category of skills is provided in the job description, the available category determines the score.

### Candidate Verdicts

| Overall Score | Verdict |
|---|---|
| 80% and above | Strong Skill Match |
| 60% – 79.99% | Moderate Skill Match |
| 40% – 59.99% | Partial Skill Match |
| Below 40% | Low Skill Match |

**Note:** Professional experience is calculated and displayed separately; it does not contribute to the weighted skill-match score.

## 🖥️ Application Interface

The application provides:

- Job description and multiple resume upload sections.
- Extracted job requirements.
- Ranked candidate dashboard.
- Required and preferred skill score breakdowns.
- Matching and missing skill lists.
- Individual candidate reports.
- Experience and project details.
- CSV export functionality.

## 📁 Project Structure

```text
ai-resume-parser/
│
├── app.py
├── matching.py
├── resume_parser.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env                 # Local API key (not committed)
```

## 🔧 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/akshatsinghrawat/ai-resume-parser.git
```

### 2. Navigate to the Project Directory

```bash
cd ai-resume-parser
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Groq API Key

Create a `.env` file in the project root directory:

```env
GROQ_API_KEY=your_groq_api_key
```

Obtain an API key from [Groq Console](https://console.groq.com/).

Never commit your `.env` file or expose your API key publicly.

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud** and connected to the GitHub repository.

Live application:

[https://akshatsinghrawat-ai-resume-parser-app-oyehwc.streamlit.app](https://akshatsinghrawat-ai-resume-parser-app-oyehwc.streamlit.app)

## ⚠️ Limitations

- The quality of extracted information depends on resume formatting and text extraction quality.
- Scanned PDFs without an extractable text layer may not be processed correctly.
- Skill matching uses predefined normalization rules and aliases; it is not full semantic similarity matching.
- Experience calculations depend on clearly stated employment or internship dates.
- The scoring formula measures skill overlap and does not independently establish a candidate's overall suitability.
- LLM-generated extraction results may require human verification.

## 🔮 Future Scope

Potential improvements include:

- Semantic skill matching using embeddings.
- More advanced experience relevance analysis.
- Support for additional resume formats.
- Enhanced candidate comparison and analytics.
- More comprehensive evaluation of skills demonstrated through projects and experience.

## 👨‍💻 Author

**Akshat Singh Rawat**

Thapar Institute of Engineering and Technology

[GitHub](https://github.com/akshatsinghrawat)

---

⭐ If you find this project interesting, feel free to explore the repository and try the live application.