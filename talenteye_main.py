#TalentEye imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from typing import Dict, Any
import re
import os
from dotenv import load_dotenv
#Loading Genai API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Binding the entire pipeline into 1 function for easy integration with Streamlit
def pipeline_talenteye_main(file_resume: str, file_job_description: str,GEMINI_API_KEY: str):
    # PDF Loading inside system
    loaded_resume = PyMuPDFLoader(file_resume)
    doc_resume = loaded_resume.load()
    loaded_job = PyMuPDFLoader(file_job_description)
    doc_job = loaded_job.load()

    # Text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 0
    )
    text_resume = doc_resume[0].page_content
    text_chunks_resume = splitter.split_text(text_resume)
    text_job = doc_job[0].page_content
    text_chunks_job = splitter.split_text(text_job)

    #Document loading
    chunks_resume = [Document(page_content=chunk) for chunk in text_chunks_resume]
    chunks_job = [Document(page_content=chunk) for chunk in text_chunks_job]

    # LLM and prompt part
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        api_key=GEMINI_API_KEY
    )
    prompt = ChatPromptTemplate(
        [
            ("system", """You are a helpful HR Assistant named 'TalentEye', you are a keen observer and can accurately decide if this Candidate is an actual match for the job he/she has 
             described in the PDF or not.
            You are being provided with 2 different documents chunked and embedded into a single 'final_docs'. It consists of a resume and job description of the person.
            Your task is to summarise the content and calculate final score of the candidate and provide it as the output in the following format:-
            Resume summary must include:
            Skills->
            Experience->
            Education->
            Achievements->

            Job description summary must include:
            Required skills->
            Experience level->
            Education requirements->
            Key responsibilities->

            Compare both summaries and calculate scores for:

            Skills - 50%
            Experience - 30%
            Education - 20%

            Compute an overall score on a 0-100 scale.
            """),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ]
    )
    final_docs = chunks_resume + chunks_job
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",
                                            google_api_key=GEMINI_API_KEY)
    vectorstore = FAISS.from_documents(final_docs, embeddings)
    retriever = vectorstore.as_retriever()
    query = "Summarize the resume and job description and provide a score based on the provided criteria also highlight the name of the person if mentioned in the content given to you."
    relevant_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    input_data = {"input": context}
    chain = prompt | llm
    response = chain.invoke(input_data)
    if isinstance(response, str):
       llm_text = response
    elif isinstance(response, dict):
       llm_text = response.get("content") or response.get("text") or str(response)
    else:
       llm_text = getattr(response, "content", None) or getattr(response, "text", None) or str(response)
    class ApplicantState(Dict[str, Any]):
        llm_response: str
        parsed_data: Dict[str, Any]
        decision: str
        process_summary: str
        final_report: str
    def parse_llm_response(content: str) -> Dict[str, Any]:
        """
        Extracts resume summary, job summary, category scores, and overall score.
        Handles MULTIPLE LLM output formats (both ** and ### headers).
        """
        data = {}

        # ========== EXTRACT RESUME SUMMARY ==========
        # Try format 1: ### Resume Summary
        resume_match = re.search(
            r"###\s*Resume\s*Summary[:\s]*\n(.*?)(?=###\s*Job\s*Description\s*Summary|###\s*Comparison|\-\-\-|\Z)",
            content,
            re.I | re.S
        )
        if not resume_match:
            # Try format 2: **Resume Summary**
            resume_match = re.search(
                r"\*\*Resume\s*Summary.*?\*\*[:\s]*\n(.*?)(?=\*\*Job\s*Description\s*Summary|###\s*Job|\-\-\-|\Z)",
                content,
                re.I | re.S
            )
        data["resume_summary"] = resume_match.group(1).strip() if resume_match else "Not found"

        # ========== EXTRACT JOB SUMMARY ==========
        # Try format 1: ### Job Description Summary
        job_match = re.search(
            r"###\s*Job\s*Description\s*Summary[:\s]*\n(.*?)(?=###\s*Comparison|\*\*Comparison|\-\-\-|\Z)",
            content,
            re.I | re.S
        )
        if not job_match:
            # Try format 2: **Job Description Summary**
            job_match = re.search(
                r"\*\*Job\s*Description\s*Summary.*?\*\*[:\s]*\n(.*?)(?=\*\*Comparison|###\s*Comparison|\-\-\-|\Z)",
                content,
                re.I | re.S
            )
        data["job_summary"] = job_match.group(1).strip() if job_match else "Not found"
        # ...existing code...
        # ...existing code...
        def extract_score(keyword: str, max_val: int, text: str) -> tuple[int, int]:
            """
            Returns (scaled_score, percent).
            - scaled_score: value on component scale (0..max_val) used for overall computation.
            - percent: interpreted percentage (0..100) if available or derived from scaled_score.
            """
            if not text:
                return 0, 0
            patterns = [
                rf"{keyword}[:\s\-–—]*([0-9]{{1,3}})\s*/\s*{max_val}",            # 45/50
                rf"{keyword}[:\s\-–—]*([0-9]{{1,3}})\s+out\s+of\s+{max_val}",     # 45 out of 50
                rf"{keyword}[:\s\-–—]*\(?([0-9]{{1,3}})\)?\s*(?:points)?",       # 45 or (45)
                rf"{keyword}.*?score[:\s]*([0-9]{{1,3}})\s*/\s*{max_val}",       # score: 45/50
                rf"{keyword}[:\s\-–—]*([0-9]{{1,3}})\s*%",                       # 90% -> percent
                rf"{keyword}[:\s\-–—]*([0-9]{{1,3}})(?!\s*%|\s*/)"               # loose number
            ]
            for p in patterns:
                m = re.search(p, text, re.I | re.S)
                if not m:
                    continue
                try:
                    val = int(m.group(1))
                except Exception:
                    continue
                # Detect explicit percent form (has % in original text near the captured number)
                pct_pattern = rf"{keyword}[:\s\-–—]*{re.escape(m.group(1))}\s*%"
                if re.search(pct_pattern, text, re.I):
                    pct = max(0, min(val, 100))
                    scaled = round(pct * max_val / 100)
                    return scaled, pct
                # If value > max_val but <=100 assume it's a percent given without % sign
                if val > max_val and val <= 100:
                    pct = val
                    scaled = round(pct * max_val / 100)
                    return scaled, pct
                # Otherwise treat as raw component score on 0..max_val
                scaled = max(0, min(val, max_val))
                pct = round(scaled * 100 / max_val) if max_val else 0
                return scaled, pct
            return 0, 0

        # use the new extractor (content is the LLM text)
        skills_score, skills_pct = extract_score("Skills", 50, content)
        exp_score, exp_pct = extract_score("Experience", 30, content)
        edu_score, edu_pct = extract_score("Education", 20, content)

        # Try to extract an explicit overall score; if missing compute from components
        overall_score = 0
        overall_patterns = [
            r"Overall\s*Score[:\s]*([0-9]{1,3})\s*/\s*100",
            r"Total.*?Score[:\s]*([0-9]{1,3})\s*/\s*100",
            r"Overall[:\s]*([0-9]{1,3})\s*%"
        ]
        for p in overall_patterns:
            m = re.search(p, content, re.I | re.S)
            if m:
                try:
                    overall_score = int(m.group(1))
                except:
                    overall_score = 0
                break
        if overall_score == 0:
            overall_score = skills_score + exp_score + edu_score
            overall_score = max(0, min(overall_score, 100))

        data["scores"] = {
            "skills": skills_score,
            "skills_pct": skills_pct,
            "experience": exp_score,
            "experience_pct": exp_pct,
            "education": edu_score,
            "education_pct": edu_pct,
            "overall": overall_score
        }
# ...existing code...
        return data

    def parse_llm_output(state: ApplicantState):
        """Node 1: Parse the LLM response"""
        state["llm_response"] = llm_text
        parsed = parse_llm_response(str(state['llm_response']))
        state["parsed_data"] = parsed
        print(f"✅ Successfully parsed. Overall score: {parsed['scores']['overall']}")
        return state


    def decision_router(state: ApplicantState):
        """Node 2: Route based on overall score - FIXED VERSION"""
        overall = state["parsed_data"]["scores"]["overall"]
        print(f"📊 Overall Score: {overall}/100")
        if overall >= 85:
            state["decision"] = "one_interview"
            print("✅ Decision: Shortlist for direct HR interview")
        elif 60 <= overall < 85:
            state["decision"] = "two_interview"
            print("⚠️  Decision: Shortlist for screening + coding interviews")
        else:
            state["decision"] = "reject"
            print("❌ Decision: Reject candidate")

        return state 


    def execute_one_interview_process(state: ApplicantState):
        """Node 3a: Execute one-interview process"""
        state["process_summary"] = "Candidate shortlisted for direct HR interview."
        return state
    def execute_two_interview_process(state: ApplicantState):
        """Node 3b: Execute two-interview process"""
        state["process_summary"] = "Candidate shortlisted for screening and coding interviews."
        return state
    def execute_rejection_process(state: ApplicantState):
        """Node 3c: Execute rejection process"""
        state["process_summary"] = "Candidate rejected after evaluation."
        return state

    def generate_final_report(state: ApplicantState):
        """Node 4: Generate final structured report - Print Ready"""
        parsed = state["parsed_data"]
        scores = parsed["scores"]
        def clean_markdown(text):
            # Normalize bullets and remove bold markers so downstream PDF uses ASCII only
            if not text:
                return ""
            text = text.replace('**', '')
            text = text.replace('•', '-')
            text = text.replace('·', '-')
            text = re.sub(r'^\s*[\*\-\u2022\u00B7]?\s*', '- ', text, flags=re.MULTILINE)
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text.strip()
        
        resume_clean = clean_markdown(parsed['resume_summary'])
        job_clean = clean_markdown(parsed['job_summary'])

        report = (
            "FINAL CANDIDATE EVALUATION REPORT\n\n"
            "RESUME SUMMARY:\n" + (resume_clean[:5000] + ("..." if len(resume_clean) > 5000 else "")) + "\n\n"
            "JOB DESCRIPTION SUMMARY:\n" + (job_clean[:5000] + ("..." if len(job_clean) > 5000 else "")) + "\n\n"
            "CATEGORY-WISE SCORES:\n"
            f"- Skills: {scores['skills']}/50  ({scores['skills']*2}%)\n"
            f"- Experience: {scores['experience']}/30  ({round(scores['experience']/30*100, 1)}%)\n"
            f"- Education: {scores['education']}/20  ({scores['education']*5}%)\n"
            f"- Overall: {scores['overall']}/100 ({scores['overall']}%)\n\n"
            "FINAL DECISION:\n" + state.get('process_summary', '') + "\n"
        )
        state["final_report"] = report
        return state
    graph = StateGraph(ApplicantState)
    graph.add_node("parse_llm_output", parse_llm_output)
    graph.add_node("decision_router", decision_router)
    graph.add_node("execute_one_interview_process", execute_one_interview_process)
    graph.add_node("execute_two_interview_process", execute_two_interview_process)
    graph.add_node("execute_rejection_process", execute_rejection_process)
    graph.add_node("generate_final_report", generate_final_report)
    graph.add_edge(START, "parse_llm_output")
    graph.add_edge("parse_llm_output", "decision_router")
    graph.add_conditional_edges("decision_router", lambda state: state["decision"], {
        "one_interview": "execute_one_interview_process",
        "two_interview": "execute_two_interview_process",
        "reject": "execute_rejection_process"
    })

    # All paths converge to final report
    graph.add_edge("execute_one_interview_process", "generate_final_report")
    graph.add_edge("execute_two_interview_process", "generate_final_report")
    graph.add_edge("execute_rejection_process", "generate_final_report")
    graph.add_edge("generate_final_report", END)

    # Compile the graph
    applicant_graph = graph.compile()
    print("✅ LangGraph workflow compiled successfully!")

    # --- Execute the workflow ---
    print("\n" + "="*80)
    print("🚀 STARTING AI HIRING RECOMMENDATION SYSTEM")
    print("="*80 + "\n")

    state = ApplicantState({"llm_response": llm_text})
    result = applicant_graph.invoke(state)
    return result