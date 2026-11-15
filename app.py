import os
import json
import re
from datetime import date
from io import BytesIO

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader
from docx import Document


# ---------------- CONFIGURATION ----------------

load_dotenv()

MODEL = "openai/gpt-oss-120b"

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
)


# ---------------- LLM CLIENT ----------------


@st.cache_resource
def get_client():
    api_key = None

    # First, check Streamlit Cloud secrets
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    # If unavailable, check the local .env file
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Configure it in Streamlit Secrets "
            "or your local .env file."
        )

    return Groq(api_key=api_key)



# ---------------- PYDANTIC SCHEMAS ----------------

class JobDescription(BaseModel):
    role: str = "Not specified"
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    education_requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)


class Experience(BaseModel):
    category: str = "other"
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    project_skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    score: float = Field(ge=0, le=100)
    required_score: float = Field(default=0, ge=0, le=100)
    preferred_score: float = Field(default=0, ge=0, le=100)
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    matching_required: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    matching_preferred: list[str] = Field(default_factory=list)
    missing_preferred: list[str] = Field(default_factory=list)
    experience_match: str = "Not assessed"
    verdict: str = ""


# ---------------- FILE EXTRACTION ----------------

def extract_text(file):
    """Extract readable text from a PDF, DOCX, or TXT upload."""
    filename = file.name.lower()
    file_bytes = file.getvalue()

    if filename.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError("This PDF is password-protected.") from exc

        extracted_pages = []
        for page in reader.pages:
            extracted_pages.append(page.extract_text() or "")
        return "\n".join(extracted_pages).strip()

    if filename.endswith(".docx"):
        document = Document(BytesIO(file_bytes))
        parts = [paragraph.text for paragraph in document.paragraphs]

        # Include text stored in tables, which often contain resume details.
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))

        return "\n".join(part for part in parts if part.strip()).strip()

    if filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="replace").strip()

    raise ValueError(
        f"Unsupported file format for {file.name}. Use PDF, DOCX, or TXT."
    )


# ---------------- LLM JSON HELPER ----------------

def ask_llm(system_prompt, user_prompt):
    """Call Groq and return the JSON response as a Python dictionary."""
    client = get_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("The AI model returned an empty response.")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("The AI model returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise ValueError("The AI response must be a JSON object.")

    return data


# ---------------- JOB DESCRIPTION PARSER ----------------

def parse_job_description(jd_text):
    schema = JobDescription.model_json_schema()

    system_prompt = f"""
You are an expert technical recruiter. Extract structured information
from the supplied job description.

Return ONLY valid JSON matching this schema:
{json.dumps(schema)}

Instructions:
1. Extract technical skills explicitly mentioned in the job description.
2. Every skill must be an individual, atomic skill. Never combine
   technologies into one item.
   Example: return ["Python", "NumPy", "Pandas", "Scikit-learn"],
   not ["Python data science libraries (NumPy, pandas, scikit-learn)"].
3. Keep distinct technologies separate, such as PyTorch and TensorFlow.
4. Put explicitly mandatory or required skills in required_skills.
5. Put explicitly preferred, advantageous, or optional skills in
   preferred_skills.
6. Normalize obvious naming variations where appropriate.
7. Do not include soft skills as technical skills.
8. Do not invent requirements. Use empty lists when information is absent.
9. Remove duplicate skills.
"""

    data = ask_llm(system_prompt, jd_text)
    return JobDescription(**data)


# ---------------- RESUME PARSER ----------------

def parse_resume(resume_text):
    schema = Resume.model_json_schema()

    system_prompt = f"""
You are an accurate resume information extraction system.

Extract candidate information from the full resume, including projects,
internships, employment and extracurricular activities. Extract technical
skills wherever they are explicitly demonstrated. Do not infer skills
merely because they are common for a role.

Return ONLY valid JSON matching this schema:
{json.dumps(schema)}

Rules:
1. Do not invent information or estimate missing details.
2. Use null for unavailable individual fields.
3. Use empty lists for unavailable list fields.
4. Extract each technical skill as a separate item, including skills
   mentioned in project descriptions and internship responsibilities.
5. Keep project descriptions concise but preserve the technologies and
   work actually stated in the resume.
6. Populate project_skills with technical skills explicitly stated in
   project descriptions. Do not infer unstated technologies.
6. For each experience, set category to one of: employment, internship,
   leadership, or other. Campus ambassador, committee, club and volunteer
   positions are leadership or other, not professional employment.
7. Do not count projects as employment.
8. total_experience_years should be a number only when it can be
   reasonably calculated from explicit employment or internship dates;
   otherwise use null.
"""

    data = ask_llm(system_prompt, resume_text)

    # Catch technologies that may be mentioned in project bullets or tables
    # but omitted from the structured response.
    extracted_skills = data.get("skills", [])
    if not isinstance(extracted_skills, list):
        extracted_skills = []
    data["skills"] = unique_preserving_order(
        extracted_skills + extract_skills_from_text(resume_text)
    )

    # Validate the extracted JSON against the Resume model before using it.
    resume = Resume(**data)

    # Supplement project_skills with technologies explicitly written in project descriptions.
    # Keep these separate from skills listed elsewhere in the resume.
    project_text = "\n".join(
        project for project in resume.projects if isinstance(project, str)
    )
    resume.project_skills = unique_preserving_order(
        resume.project_skills + extract_skills_from_text(project_text)
    )

    calculated_experience = calculate_professional_experience(resume.experiences)
    if calculated_experience is not None:
        resume.total_experience_years = calculated_experience
    return resume


# ---------------- CANDIDATE MATCHING ----------------

SKILL_ALIASES = {
    "python programming": "python",
    "core python": "python",
    "object oriented programming": "oop",
    "object oriented programming oop": "oop",
    "oops": "oop",
    "database management systems": "dbms",
    "database management systems dbms": "dbms",
    "machine learning fundamentals": "machine learning",
    "ml": "machine learning",
    "deep learning framework": "deep learning",
    "sklearn": "scikit learn",
    "scikit learn": "scikit learn",
    "structured query language": "sql",
    "rest apis": "rest api",
    "restful api": "rest api",
    "restful apis": "rest api",
    "api development": "rest api",
    "javascript programming": "javascript",
    "c plus plus": "c++",
    "c plus plus programming": "c++",
    "java programming": "java",
    "dsa": "data structures and algorithms",
    "data structures algorithms": "data structures and algorithms",
    "data structures and algorithm": "data structures and algorithms",
}

# This list supplements the model's extraction with technologies that are
# explicitly present in the source text. It is deliberately conservative.
KNOWN_SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C", "C#",
    "SQL", "Advanced SQL", "MySQL", "PostgreSQL", "Oracle", "SQLite",
    "MongoDB", "Snowflake", "DBMS", "RDBMS", "NumPy", "Pandas",
    "Matplotlib", "Seaborn", "Plotly", "Scikit-learn", "TensorFlow",
    "PyTorch", "Keras", "OpenCV", "YOLOv8", "ONNX", "Power BI",
    "Tableau", "Sigma Computing", "Excel", "DAX", "Streamlit",
    "Flask", "FastAPI", "Django", "React", "Node.js", "HTML", "CSS",
    "Git", "GitHub", "Docker", "AWS", "Azure", "GCP", "Linux",
    "REST API", "Machine Learning", "Deep Learning", "NLP",
    "Statistics", "Probability", "Linear Algebra", "Data Structures",
    "Algorithms", "Object-Oriented Programming", "OOP", "Data Analysis",
    "Data Cleaning", "Data Visualization", "Logistic Regression",
    "Random Forest", "XGBoost", "Predictive Modeling", "EDA",
]


def normalize_skill(skill):
    """Standardise common skill names without relying on fuzzy matching."""
    if not isinstance(skill, str):
        return ""

    normalized = skill.lower().strip().replace("&", " and ")
    normalized = normalized.replace("scikit-learn", "scikit learn")
    normalized = normalized.replace("node.js", "node js")
    normalized = re.sub(r"[^a-z0-9+#]+", " ", normalized)
    normalized = " ".join(normalized.split())
    return SKILL_ALIASES.get(normalized, normalized)


def extract_skills_from_text(text):
    """Find known skills that are explicitly written in the resume text."""
    found = []
    source = text.lower()

    for skill in sorted(KNOWN_SKILLS, key=len, reverse=True):
        pattern = r"(?<![a-z0-9+#])" + re.escape(skill.lower()) + r"(?![a-z0-9+#])"
        if re.search(pattern, source):
            found.append(skill)

    return unique_preserving_order(found)


def find_skill_evidence(skill, resume):
    """Return the resume sections where a skill is explicitly mentioned."""
    target = normalize_skill(skill)
    evidence = []

    if any(normalize_skill(item) == target for item in resume.skills):
        evidence.append("Skills section")

    if any(
        normalize_skill(item) == target
        for experience in resume.experiences
        for item in experience.skills_used
    ):
        evidence.append("Experience")

    if any(normalize_skill(item) == target for item in resume.project_skills):
        evidence.append("Projects")

    # The structured parser may not always assign every explicit skill to
    # a section. Avoid inventing a location when no section-level evidence exists.
    return evidence


def skill_matches(required_skill, candidate_skills):
    required = normalize_skill(required_skill)
    if not required:
        return False

    candidate_names = [normalize_skill(item) for item in candidate_skills]
    if required in candidate_names:
        return True

    # A resume may list the combined phrase while the JD lists its parts.
    compound = "data structures and algorithms"
    if compound in candidate_names and required in {"data structures", "algorithms"}:
        return True
    if required == compound and {
        "data structures", "algorithms"
    }.issubset(set(candidate_names)):
        return True

    # Do not treat MySQL/PostgreSQL as proof of SQL knowledge. An explicit
    # SQL phrase (including Advanced SQL) is required for an SQL match.
    if required == "sql":
        return any(name in {"sql", "advanced sql"} for name in candidate_names)
    if required == "advanced sql":
        return "advanced sql" in candidate_names

    return False


def _experience_months(duration):
    """Return full months from a written date range, if one can be read."""
    if not duration or not isinstance(duration, str):
        return None

    full_months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    month_names = {}
    for number, name in enumerate(full_months, 1):
        month_names[name.lower()] = number
        month_names[name[:3].lower()] = number
    month_pattern = "|".join(sorted(month_names, key=len, reverse=True))
    match = re.search(
        rf"({month_pattern})\s+(20\d{{2}})\s*(?:-|–|—|to)\s*"
        rf"({month_pattern})\s+(20\d{{2}})",
        duration.lower(),
    )
    if match:
        start_month = month_names[match.group(1)]
        start_year = int(match.group(2))
        end_month = month_names[match.group(3)]
        end_year = int(match.group(4))
        months = (end_year - start_year) * 12 + end_month - start_month + 1
        return months if months > 0 else None

    return None


def calculate_professional_experience(experiences):
    """Add explicit internship/employment durations; exclude leadership roles."""
    total_months = 0
    found_duration = False

    for item in experiences:
        category = (item.category or "other").lower()
        if category not in {"employment", "internship"}:
            continue
        months = _experience_months(item.duration)
        if months is not None:
            total_months += months
            found_duration = True

    if not found_duration:
        return None

    return round(total_months / 12, 2)


def unique_preserving_order(items):
    """Remove duplicates while retaining the original order."""
    seen = set()
    unique_items = []

    for item in items:
        if not isinstance(item, str) or not item.strip():
            continue

        key = normalize_skill(item)
        if key and key not in seen:
            seen.add(key)
            unique_items.append(item.strip())

    return unique_items


def evaluate_candidate(job, resume):
    candidate_skills = list(resume.skills)

    for experience in resume.experiences:
        candidate_skills.extend(experience.skills_used)

    candidate_skills.extend(resume.project_skills)
    candidate_skills = unique_preserving_order(candidate_skills)

    required = unique_preserving_order(job.required_skills)
    preferred = unique_preserving_order(job.preferred_skills)

    matching_required = [
        skill for skill in required
        if skill_matches(skill, candidate_skills)
    ]
    missing_required = [
        skill for skill in required
        if not skill_matches(skill, candidate_skills)
    ]
    matching_preferred = [
        skill for skill in preferred
        if skill_matches(skill, candidate_skills)
    ]
    missing_preferred = [
        skill for skill in preferred
        if not skill_matches(skill, candidate_skills)
    ]

    required_score = (
        len(matching_required) / len(required) * 100 if required else 0
    )
    preferred_score = (
        len(matching_preferred) / len(preferred) * 100 if preferred else 0
    )

    # Keep the project's documented 80:20 required/preferred weighting.
    if required and preferred:
        final_score = required_score * 0.8 + preferred_score * 0.2
    elif required:
        final_score = required_score
    elif preferred:
        final_score = preferred_score
    else:
        final_score = 0

    if final_score >= 80:
        verdict = "Strong skill match"
    elif final_score >= 60:
        verdict = "Moderate skill match"
    elif final_score >= 40:
        verdict = "Partial skill match"
    else:
        verdict = "Low skill match"

    experience_match = (
        f"{resume.total_experience_years:g} years reported"
        if resume.total_experience_years is not None
        else "Experience information unavailable"
    )

    return MatchResult(
        score=round(final_score, 2),
        required_score=round(required_score, 2),
        preferred_score=round(preferred_score, 2),
        matching_skills=matching_required + matching_preferred,
        missing_skills=missing_required + missing_preferred,
        matching_required=matching_required,
        missing_required=missing_required,
        matching_preferred=matching_preferred,
        missing_preferred=missing_preferred,
        experience_match=experience_match,
        verdict=verdict,
    )


# ---------------- USER INTERFACE ----------------

st.title("📄 AI Resume Parser & Candidate Ranking")
st.caption("Recruitment screening with structured resume extraction and explainable skill matching")

st.info(
    "Upload a job description and multiple resumes to analyse "
    "candidate suitability. Scores represent skill overlap, not a "
    "final hiring decision. Review the resume evidence before making any recruitment decision."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Job Description")
    jd_file = st.file_uploader(
        "Upload JD (PDF, DOCX or TXT)",
        type=["pdf", "docx", "txt"],
        key="jd_upload",
    )

with col2:
    st.subheader("2. Candidate Resumes")
    resume_files = st.file_uploader(
        "Upload resumes (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="resume_uploads",
    )

if st.button("Analyse Candidates", type="primary"):
    if not jd_file or not resume_files:
        st.error("Please upload a job description and at least one resume.")
    else:
        try:
            jd_text = extract_text(jd_file)
            if not jd_text:
                st.error("No readable text found in the job description.")
                st.stop()

            with st.spinner("Analysing job description with AI..."):
                job = parse_job_description(jd_text)

            st.success(f"Job requirements extracted: {job.role}")

            results = []
            progress = st.progress(0)
            status = st.empty()

            for index, file in enumerate(resume_files):
                status.write(f"Processing: {file.name}")

                try:
                    resume_text = extract_text(file)
                    if not resume_text:
                        raise ValueError("No readable text found in this file.")

                    parsed_resume = parse_resume(resume_text)
                    match = evaluate_candidate(job, parsed_resume)

                    results.append({
                        "Name": parsed_resume.name or file.name,
                        "Email": parsed_resume.email or "Not provided",
                        "Score": match.score,
                        "Required Score": match.required_score,
                        "Preferred Score": match.preferred_score,
                        "Matching Required Skills": ", ".join(match.matching_required),
                        "Missing Required Skills": ", ".join(match.missing_required),
                        "Matching Preferred Skills": ", ".join(match.matching_preferred),
                        "Missing Preferred Skills": ", ".join(match.missing_preferred),
                        "Matching Skills": ", ".join(match.matching_skills),
                        "Missing Skills": ", ".join(match.missing_skills),
                        "Experience Match": match.experience_match,
                        "Verdict": match.verdict,
                        "Skills": ", ".join(parsed_resume.skills),
                        "Project Skills": ", ".join(parsed_resume.project_skills),
                        "Education": ", ".join(parsed_resume.education),
                        "Experience Details": " | ".join(
                            f"{item.role or 'Role not specified'} at {item.company or 'Organisation not specified'} ({item.category})"
                            for item in parsed_resume.experiences
                        ),
                        "Projects": " | ".join(parsed_resume.projects),
                        "Certifications": " | ".join(parsed_resume.certifications),
                        "Resume": file.name,
                        "Profile": parsed_resume.model_dump(),
                    })

                except (ValidationError, ValueError) as exc:
                    st.warning(f"Could not process {file.name}: {exc}")
                except Exception as exc:
                    st.warning(
                        f"An unexpected error occurred while processing "
                        f"{file.name}: {exc}"
                    )

                progress.progress((index + 1) / len(resume_files))

            status.empty()

            if not results:
                st.error("No resumes could be processed successfully.")
                st.stop()

            results.sort(key=lambda item: item["Score"], reverse=True)
            st.session_state["results"] = results
            st.session_state["job_requirements"] = job.model_dump()

        except Exception as exc:
            st.error(f"Application error: {exc}")


# ---------------- RESULTS DASHBOARD ----------------

if "results" in st.session_state:
    results = st.session_state["results"]
    job_data = st.session_state.get("job_requirements", {})

    st.divider()
    st.header("Candidate Ranking")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Candidates Processed", len(results))
    metric2.metric("Highest Match", f"{results[0]['Score']:.2f}%")
    average_score = sum(item["Score"] for item in results) / len(results)
    metric3.metric("Average Match", f"{average_score:.1f}%")

    with st.expander("View extracted job requirements"):
        st.write("**Role:**", job_data.get("role", "Not specified"))
        st.write(
            "**Required Skills:**",
            ", ".join(job_data.get("required_skills", [])) or "None extracted",
        )
        st.write(
            "**Preferred Skills:**",
            ", ".join(job_data.get("preferred_skills", [])) or "None extracted",
        )
        st.write(
            "**Education Requirements:**",
            ", ".join(job_data.get("education_requirements", [])) or "Not specified",
        )

    st.subheader("Ranked Candidates")

    display_data = [
        {
            "Rank": index + 1,
            "Candidate": result["Name"],
            "Required Skills": f"{result['Required Score']:.2f}%",
            "Preferred Skills": f"{result['Preferred Score']:.2f}%",
            "Overall Score": f"{result['Score']:.2f}%",
            "Verdict": result["Verdict"],
        }
        for index, result in enumerate(results)
    ]

    st.dataframe(
        pd.DataFrame(display_data),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Individual Candidate Reports")

    for index, result in enumerate(results):
        with st.expander(
            f"#{index + 1} | {result['Name']} | {result['Score']:.2f}%"
        ):
            st.write("**Email:**", result["Email"])
            st.write("**Resume:**", result["Resume"])

            st.markdown("#### Score Breakdown")
            score_col1, score_col2, score_col3 = st.columns(3)

            score_col1.metric(
                "Required Skills",
                f"{result['Required Score']:.2f}%",
            )
            score_col2.metric(
                "Preferred Skills",
                f"{result['Preferred Score']:.2f}%",
            )
            score_col3.metric(
                "Overall Match",
                f"{result['Score']:.2f}%",
            )

            profile_skills_for_report = (
                result["Profile"].get("skills", [])
                + [
                    used_skill
                    for experience in result["Profile"].get("experiences", [])
                    for used_skill in experience.get("skills_used", [])
                ]
            )
            matching_required_for_report = [
                skill for skill in job_data.get("required_skills", [])
                if skill_matches(skill, profile_skills_for_report)
            ]
            matching_preferred_for_report = [
                skill for skill in job_data.get("preferred_skills", [])
                if skill_matches(skill, profile_skills_for_report)
            ]
            missing_required_for_report = [
                skill for skill in job_data.get("required_skills", [])
                if not skill_matches(skill, profile_skills_for_report)
            ]
            missing_preferred_for_report = [
                skill for skill in job_data.get("preferred_skills", [])
                if not skill_matches(skill, profile_skills_for_report)
            ]
            st.write("**Matching Required Skills:**", ", ".join(matching_required_for_report) or "None")
            st.write("**Missing Required Skills:**", ", ".join(missing_required_for_report) or "None")
            st.write("**Matching Preferred Skills:**", ", ".join(matching_preferred_for_report) or "None")
            st.write("**Missing Preferred Skills:**", ", ".join(missing_preferred_for_report) or "None")
            st.write("**Experience:**", result["Experience Match"])
            st.write("**Verdict:**", result["Verdict"])
            st.write("**All Extracted Skills:**", result["Skills"] or "None")
            st.write("**Project Skills:**", result.get("Project Skills", "Not listed"))
            st.write("**Education:**", result["Education"] or "Not provided")

            profile = result["Profile"]

            st.markdown("#### Evidence for Matched Required Skills")
            matched_items = match_required = [
                item.strip()
                for item in result["Matching Required Skills"].split(",")
                if item.strip() and item.strip() != "None"
            ]
            if matched_items:
                for skill in matched_items:
                    locations = find_skill_evidence(skill, Resume(**profile))
                    st.write(
                        f"- **{skill}:** {', '.join(locations) if locations else 'Extracted by resume parser; section not identified'}"
                    )
            else:
                st.caption("No required skills were matched.")

            if profile.get("experiences"):
                st.markdown("#### Experience Details")
                for experience in profile["experiences"]:
                    heading = " — ".join(
                        value for value in [
                            experience.get("role"),
                            experience.get("company"),
                        ] if value
                    )
                    category = (experience.get("category") or "other").replace("_", " ").title()
                    st.write(f"**{heading or 'Experience'}** · {category}")
                    if experience.get("duration"):
                        st.caption(experience["duration"])
                    if experience.get("description"):
                        st.write(experience["description"])

            if profile.get("projects"):
                st.markdown("#### Projects")
                for project in profile["projects"]:
                    st.write(f"- {project}")

            if profile.get("certifications"):
                st.markdown("#### Certifications")
                for certification in profile["certifications"]:
                    st.write(f"- {certification}")

            st.progress(result["Score"] / 100)

    # Export results without internal profile data.
    export_data = [
        {key: value for key, value in result.items() if key != "Profile"}
        for result in results
    ]
    csv_data = pd.DataFrame(export_data).to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Candidate Results (CSV)",
        data=csv_data,
        file_name="candidate_ranking.csv",
        mime="text/csv",
    )

    if st.button("Clear Results"):
        st.session_state.pop("results", None)
        st.session_state.pop("job_requirements", None)
        st.rerun()
