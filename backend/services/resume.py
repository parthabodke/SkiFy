import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from services.gemini import call_openrouter

load_dotenv()

def extract_pdf_content(pdf_bytes: bytes) -> str:
    """
    Extracts text content from all pages of a PDF.
    """
    all_text_content = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            all_text_content += page.get_text("text") + "\n\n---PAGE BREAK---\n\n"
        doc.close()
        return all_text_content
    except Exception as e:
        raise Exception(f"Error processing PDF with PyMuPDF: {e}")

def get_resume_analysis(job_description: str, pdf_bytes: bytes, analysis_type: str) -> str:
    text_content = ""
    if analysis_type in ["summary", "percentage_match"]:
        text_content = extract_pdf_content(pdf_bytes)

    if analysis_type == "summary":
        prompt = f"""
        You are an elite Principal Technical Recruiter and Engineering Manager reviewing a candidate's resume.
        Provide a highly thorough, professional evaluation of how well the candidate's skills and experience align with the target role.

        Job Description:
        {job_description}

        Resume Extracted Text:
        {text_content}

        ---
        The evaluation must be highly descriptive, technically precise, and sound entirely natural and human-authored, completely avoiding typical robotic, overly polite, or generic "AI-generated" phrasing.
        Do not use any emojis or icons of any kind. Maintain a strictly professional, formal, and authoritative editorial tone.

        Structure your analysis using the following professional headings:

        ### Resume Evaluation Report

        #### 1. Content & Experience Alignment
        Perform a deep comparison of the candidate's skills and work history against the requirements:
        - **High-Alignment Areas**: List specific libraries, frameworks, or architectural accomplishments from the resume that directly match key job requirements.
        - **Crucial Gaps**: Identify essential skills, tools, or levels of experience required in the job description that are missing or weak in the resume.

        #### 2. Visual Structure & Readability Analysis
        Evaluate the presentation of the resume's text content:
        - **Information Flow**: Is the experience presented chronologically and effectively? Does it highlight results (e.g., using STAR method with metrics) or is it just a list of responsibilities?
        - **Readability & Structure**: Evaluate the clarity of the text layout, font hierarchy, and bullet point effectiveness for technical review.

        #### 3. Technical Strengths & Weaknesses Matrix
        - **Key Strengths**: Cite specific, evidence-backed achievements from the resume.
        - **Areas of Concern**: Explain what qualifications might be a bottleneck or risk factor for this specific role.

        #### 4. Strategic Recommendations
        Provide 3 highly actionable, concrete bullet points on how the candidate can optimize their resume to better target this specific role (e.g., specific wording, highlighting certain projects, or restructuring).
        """
        response_text = call_openrouter(prompt)

    elif analysis_type == "percentage_match":
        prompt = f"""
        You are an expert ATS (Applicant Tracking System) Auditor and Data Scientist specializing in resume parsing algorithms.
        Evaluate the candidate's resume text against the job description for search-relevancy and parsing compatibility.

        Job Description:
        {job_description}

        Resume Extracted Text:
        {text_content}

        ---
        The evaluation must be highly descriptive, technically precise, and sound entirely natural and human-authored, completely avoiding typical robotic, overly polite, or generic "AI-generated" phrasing.
        Do not use any emojis or icons of any kind. Maintain a strictly professional, formal, and authoritative editorial tone.

        Calculate the match percentage mathematically based on the following algorithm:
        - 60% Weight: Direct match of primary hard skills and technologies.
        - 30% Weight: Core experience alignment (role depth, senior keywords, domain exposure).
        - 10% Weight: Secondary frameworks, methodologies (Agile, CI/CD), and soft skills.

        Format your output strictly using this structure:

        Percentage Match: [XX]%

        Missing Keywords:
        - **Primary Technologies**: [Technology 1], [Technology 2] (Crucial tech stack missing from the resume)
        - **Methodologies & Processes**: [Methodology 1] (Agile, TDD, CI/CD, etc. missing)
        - **Secondary Skills**: [Skill 1] (Secondary requirements missing)

        Final Thoughts:
        [Write a concise, professional, and technical evaluation (3-4 sentences) analyzing the resume's ATS parser compatibility. Detail whether the visual text structure, section naming conventions, and chronological flow will facilitate or hinder automatic parsing, and how formatting choices affect search scores.]
        """
        response_text = call_openrouter(prompt)

    elif analysis_type == "top_skills":
        prompt = f"""
        You are a high-speed parsing agent. Extract the top 3 most important technical skill keywords required in the following job description.

        Job Description:
        {job_description}

        ---
        [RULES]
        - Extract ONLY standard, clean technology nouns (e.g., "Python", "React", "Docker", "FastAPI"). Do not extract verbs or adjectives.
        - Strictly return a valid JSON object matching the schema below.
        - Do NOT include any markdown formatting, backticks, or intro/outro text. Return only the raw JSON.

        {{"skill_1": "PrimarySkill", "skill_2": "SecondarySkill", "skill_3": "TertiarySkill"}}
        """
        response_text = call_openrouter(prompt, json_mode=True)

    else:
        raise ValueError("Invalid analysis_type. Must be 'summary', 'percentage_match', or 'top_skills'.")

    return response_text