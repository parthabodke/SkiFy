import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
import json
import re
import os
import httpx
import time
from dotenv import load_dotenv
from services.rag_service import retrieve_context 

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

def call_openrouter(prompt: str, json_mode: bool = False) -> str:
    """
    Calls OpenRouter chat completions API with the configured model and prompt.
    Supports JSON output mode.
    """
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        raise ValueError("OPENROUTER_API_KEY is not set or is still the placeholder. Please configure it in your .env file.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "SkiFy"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7
    }

    if json_mode:
        payload["response_format"] = { "type": "json_object" }

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    print(f"[-] OpenRouter HTTP {response.status_code} Error Response: {response.text} (Attempt {attempt}/{max_retries})")
                response.raise_for_status()
                data = response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                else:
                    error_msg = data.get("error", {}).get("message", "Unknown error from OpenRouter")
                    raise Exception(f"OpenRouter Error: {error_msg}")
        except Exception as e:
            print(f"[-] OpenRouter Request Failed (Attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                sleep_time = 1.5 * attempt
                print(f"[*] Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                raise e

def clamp(time_ratio):
    if time_ratio > 1.1: return 1.2
    elif 0.9 <= time_ratio <= 1.1: return 1.0
    else: return 0.8

def adjust_difficulty(current_level, window):
    user_score = 0
    for q in window:
        if q["correct"]:
            time_ratio = q["expected_time"] / q["time_taken"]
            t_bonus = clamp(time_ratio)
            user_score += q["difficulty"] * t_bonus

    upper_bound = sum([q["difficulty"] * 1.2 for q in window])
    if upper_bound == 0: return current_level
    
    performance = (user_score / upper_bound) * 100
    if performance >= 70: return min(current_level + 1, 5)
    elif performance <= 40: return max(current_level - 1, 1)
    else: return current_level

def clean_output(text):
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

def generate_question(topic, difficulty, previous_questions=[]):
    # RAG Retrieval
    context_fact = retrieve_context(topic)
    
    # Exclusion List
    avoid_list = [q.get("question", "")[:50] for q in previous_questions]
    avoid_instruction = ""
    if avoid_list:
        avoid_instruction = f"CONSTRAINT: Do NOT generate a question similar to: {json.dumps(avoid_list)}"

    # Construct Prompt
    context_instruction = ""
    if context_fact:
        print(f"[+] RAG Active for '{topic}'")
        context_instruction = f"""
        CONTEXT SEED: "{context_fact}"
        INSTRUCTION: Use the technical concept in the SEED as the core topic.
        Even if the requested difficulty is LOW (Level 1-2), do NOT ignore this seed.
        Instead, SIMPLIFY the concept. Ask a basic definition or identification question about this specific seed.
        """
    else:
        print(f"[!] RAG Bypassed for '{topic}'")
        context_instruction = f"""
        INSTRUCTION: Generate a unique question on {topic}. {avoid_instruction}
        """

    prompt = f"""
    You are a Principal Software Engineer and Technical Educator designing a professional certification quiz.
    Your goal is to generate exactly 1 multiple-choice question on the topic: "{topic}" at a strict difficulty level of {difficulty}/5.

    The question and options must be highly descriptive, technically precise, and sound entirely natural and human-authored, avoiding typical robotic or generic "AI-generated" phrasing. 
    Do not use any emojis or icons.

    ---
    [TECHNICAL DIFFICULTY SCALER]
    Level 1 (Basic/Conceptual): Core definitions, key terminologies, and basic syntax/use cases.
    Level 2 (Application): Intermediate syntax, standard library features, and simple code comprehension.
    Level 3 (Scenarios): Debugging common exceptions, practical development trade-offs, and multi-component usage.
    Level 4 (Advanced): Advanced optimizations, concurrency patterns, security implications, and design trade-offs.
    Level 5 (Expert/Internal): Deep internals, low-level compiler/runtime mechanics, micro-optimizations, and expert-level system architecture.

    {context_instruction}
    {avoid_instruction}

    ---
    [QUESTION GENERATION MANDATES]
    - **Conceptual Depth**: The question must test real comprehension, not rote memorization of simple facts.
    - **Plausible Distractors**: All 4 options must be highly plausible, technically sound, and equal in length/grammar. Avoid obviously joke options or "all of the above" / "none of the above" styles.
    - **No Giveaways**: Do not use key words in the correct option that are directly taken from the question stem.
    - **Strict Time Calibration**: Align the expected time precisely with complexity:
      - Level 1: 10-15s
      - Level 2: 15-30s
      - Level 3: 30-50s
      - Level 4: 50-80s
      - Level 5: 80-120s

    ---
    [REQUIRED JSON FORMAT]
    Return ONLY a valid JSON object matching this exact schema:
    {{
      "question": "Clear, technically precise question statement",
      "options": [
        "Option A (Index 0)",
        "Option B (Index 1)",
        "Option C (Index 2)",
        "Option D (Index 3)"
      ],
      "correct_option": 0,
      "expected_time_sec": 30
    }}

    *Rule: "correct_option" must be a strict integer (0, 1, 2, or 3) representing the index of the correct answer. Do not wrap JSON in markdown ticks.*
    """
    
    try:
        response_text = call_openrouter(prompt, json_mode=True)
        if not response_text:
            raise ValueError("OpenRouter returned an empty or null response.")
        cleaned = clean_output(response_text)
        return json.loads(cleaned)
    except Exception as e:
        print(f"Error generating question: {e}")
        return {
            "question": f"Error generating question for {topic}.",
            "options": ["Error", "Error", "Error", "Error"],
            "correct_option": "0",
            "expected_time_sec": 30
        }

def calculate_final_score(all_questions):
    if not all_questions: return 0
    user_score = 0
    for q in all_questions:
        if q["correct"]:
            time_ratio = q["expected_time"] / q["time_taken"]
            t_bonus = clamp(time_ratio) 
            user_score += q["difficulty"] * t_bonus
    upper_bound = sum([q["difficulty"] * 1.2 for q in all_questions])
    if upper_bound == 0: return 0
    return round((user_score / upper_bound) * 100)

def generate_holistic_report(all_test_results, job_description_text):
    correct_count = sum(1 for q in all_test_results if q["correct"])
    total_count = len(all_test_results)
    avg_difficulty = sum(q['difficulty'] for q in all_test_results) / total_count if total_count > 0 else 0
    
    results_summary = f"Score: {correct_count}/{total_count}, Avg Difficulty: {avg_difficulty:.1f}/5"

    prompt = f"""
    You are a Principal Engineering Manager and Technical Recruiter conducting a final candidate debrief. 
    Analyze the candidate's performance on the technical assessment against the Job Description.

    Job Description:
    {job_description_text}

    Assessment Performance Summary:
    {results_summary}

    ---
    Generate a highly structured, professional, and visually clear holistic evaluation report using clean Markdown. 
    The report must be highly descriptive, technically precise, and sound entirely natural and human-authored, completely avoiding typical robotic, overly polite, or generic "AI-generated" phrasing.
    Do not use any emojis or icons of any kind. Maintain a strictly professional, formal, and authoritative editorial tone.

    Format the report using the following structure:

    ### Holistic Assessment Report

    #### 1. Executive Summary
    Provide a high-level, 3-4 sentence professional debrief of the candidate's performance, outlining their general competency level and overall readiness for the role.

    #### 2. Core Technical Strengths
    - **[Strength 1]**: Detailed technical description of what they did well, citing specific competency evidence from the test results.
    - **[Strength 2]**: Another key area of high performance (e.g. speed, accuracy under pressure, or problem-solving capability).

    #### 3. Priority Areas for Growth
    - **[Gap 1]**: Identify a specific skill or conceptual gap indicated by their performance, explaining why it matters for this role.
    - **[Gap 2]**: Describe another technical area needing development, outlining the potential impact if left unaddressed.

    #### 4. Actionable 30-60-90 Day Growth Plan
    - **Days 1–30**: Immediate conceptual leveling up (specify topics, libraries, or frameworks to master).
    - **Days 31–60**: Practical project exposure (suggest building or refactoring components related to their gaps).
    - **Days 61–90**: Advanced system design and optimization goals.

    #### 5. Role Alignment & Job Fit
    - **Role Fit Score**: [Provide a brief score description, e.g., "Highly Aligned", "Strong Core, Tech Gaps", "Needs Development"]
    - **Final Recommendation**: Clear, constructive technical advice for the candidate's next career steps.
    """
    try:
        response_text = call_openrouter(prompt, json_mode=False)
        return response_text
    except Exception as e:
        return f"Error generating report: {e}"