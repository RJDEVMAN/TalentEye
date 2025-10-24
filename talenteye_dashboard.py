import streamlit as st
from talenteye_main import pipeline_talenteye_main
import tempfile,os
from fpdf import FPDF
import io
import re
# Main App dashboard
st.set_page_config(page_title="TalentEye", layout="wide")
st.title("📄 TalentEye — Where Talent🦅 meets Recognition👁️")
gif_url = "https://cdn.dribbble.com/userupload/23731317/file/original-5eb2f9967073700b38a31280cc2c32e0.gif"
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{gif_url}");
        background-size: cover;
    }}
    .stHeading h1 {{
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{gif_url}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.set_page_config(page_title="TalentEye", layout="wide", page_icon="👁️")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
if not GEMINI_API_KEY:
    st.warning("⚠️ Gemini API key not found. Please add it in .streamlit/secrets.toml")

resume_file = st.file_uploader("Upload Candidate Resume (PDF)", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

if st.button("🚀 Run the Evaluation", use_container_width=True):
    if not resume_file or not jd_file:
        st.error("Please upload both resume and job description files.")
    else:
        with st.spinner("Analyzing and evaluating candidate..."):
            # Save uploaded files temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_resume:
                tmp_resume.write(resume_file.read())
                resume_path = tmp_resume.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_jd:
                tmp_jd.write(jd_file.read())
                jd_path = tmp_jd.name

            result = pipeline_talenteye_main(resume_path, jd_path, GEMINI_API_KEY)

            os.remove(resume_path)
            os.remove(jd_path)

        st.success("✅ Evaluation complete!")
        report = result["final_report"]
        parsed = result["parsed_data"]
        scores = parsed["scores"]
        def clean_markdown(text):
            # Remove ** symbols
            text = text.replace('**', '')
            # Remove * at the start of lines (bullet points)
            text = re.sub(r'^\s*\*\s*', '  - ', text, flags=re.MULTILINE)
            return text.strip()
        resume_clean = clean_markdown(parsed['resume_summary'])
        job_clean = clean_markdown(parsed['job_summary'])
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "FINAL CANDIDATE EVALUATION REPORT", ln=True, align="C")

        pdf.set_font("Arial", "B", 12)
        pdf.ln(5)
        pdf.cell(0, 10, "RESUME SUMMARY", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, resume_clean[:5000] + ("..." if len(resume_clean) > 5000 else ""))

        pdf.set_font("Arial", "B", 12)
        pdf.ln(3)
        pdf.cell(0, 10, "JOB DESCRIPTION SUMMARY", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, job_clean[:5000] + ("..." if len(job_clean) > 5000 else ""))

        pdf.set_font("Arial", "B", 12)
        pdf.ln(3)
        pdf.cell(0, 10, "CATEGORY-WISE SCORES", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8,
            f"- Skills: {scores['skills']}/50  ({scores['skills']*2}%)\n"
            f"- Experience: {scores['experience']}/30  ({round(scores['experience']/30*100, 1)}%)\n"
            f"- Education: {scores['education']}/20  ({scores['education']*5}%)\n"
            f"- Overall: {scores['overall']}/100 ({scores['overall']}%)"
        )
        pdf.set_font("Arial", "B", 12)
        pdf.ln(3)
        pdf.cell(0, 10, "FINAL DECISION", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, result['process_summary'])
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        pdf_buffer = io.BytesIO(pdf_bytes)
        pdf_buffer.seek(0)
        st.subheader("📊 Candidate Evaluation Summary")
        st.metric("Overall Score", f"{scores['overall']}/100")

        col1, col2, col3 = st.columns(3)
        col1.metric("Skills", f"{scores['skills']}/50")
        col2.metric("Experience", f"{scores['experience']}/30")
        col3.metric("Education", f"{scores['education']}/20")

        st.subheader("🧠 Final Decision")
        st.info(result["process_summary"])

        st.subheader("📋 Detailed Report")
        st.text_area("Final Report", report, height=400)
        st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="Candidate_Evaluation_Report.pdf",
        mime="application/pdf"
        )