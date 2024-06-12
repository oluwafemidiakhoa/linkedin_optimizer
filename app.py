import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from docx import Document
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
access_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to generate a prompt template
def generate_prompt(profile_text, section):
    template = f"""
    You are an AI designed to read, analyze, and provide optimization suggestions for LinkedIn profiles.
    Given a LinkedIn profile, generate a detailed and professional report tailored to the profile owner.
    The report should assess the current optimization level of each section in percentage and provide actionable suggestions for improvement.
    The report should be structured clearly, with specific, actionable suggestions for improvement in the following section:

    {section}
    Please provide how much it's optimized already.

    Extracted LinkedIn Profile:
    {profile_text}
    """
    return template

# Function to get response from the LLM
def get_llm_response(prompt, profile_text):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_new_tokens=2048, temperature=0.5, huggingfacehub_api_token=access_token,
    )
    prompt_template = PromptTemplate.from_template(prompt)
    llm_chain = prompt_template | llm
    response = llm_chain.invoke({"profile_text": profile_text})
    return response

# Function to save content as a text file
def save_as_text(content):
    return BytesIO(content.encode('utf-8'))

# Function to save content as a Word document
def save_as_word(content):
    doc = Document()
    doc.add_paragraph(content)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Function to generate suggestions for a specific section
def generate_suggestions(text, section):
    st.subheader(f"{section} Suggestions")
    with st.spinner(f"Generating {section} Suggestions..."):
        prompt = generate_prompt(text, section)
        suggestions = get_llm_response(prompt, text)
        st.text_area(f"{section} Suggestions", suggestions, height=300)
        # You might want to calculate and display optimization percentage here
        # optimization_percentage = calculate_optimization_percentage(text, suggestions)
        # st.progress(optimization_percentage)

# Main function
def main():
    st.set_page_config(page_title="LinkedIn Profile Optimizer", page_icon="📄", layout="centered")
    st.title("📄 AI LinkedIn Profile Optimizer")
    st.markdown("""
    ### Welcome to the LinkedIn Profile Optimizer!
    Upload a PDF of your LinkedIn profile, and we’ll analyze it to provide you with a comprehensive summary and suggestions for improvement.
    - The report will include analysis and suggestions for better keyword usage, content formatting, and section improvements.
    - You’ll receive clear, actionable steps to enhance your LinkedIn profile.
    """)

    pdf_file = st.file_uploader("Upload your LinkedIn profile in PDF format", type=["pdf"])
    if pdf_file is not None:
        st.info("File uploaded successfully. Extracting text...")
        pdf_bytes = BytesIO(pdf_file.read())
        text = extract_text_from_pdf(pdf_bytes)

        st.subheader("Extracted Text")
        st.text_area("Extracted Text", text, height=300)

        if st.button("Get Headline Enhancement Suggestions"):
            generate_suggestions(text, "Headline Enhancement")
        if st.button("Get Summary Optimization Suggestions"):
            generate_suggestions(text, "Summary Optimization")
        if st.button("Get Experience Section Suggestions"):
            generate_suggestions(text, "Experience Section")
        
        st.download_button("Download Suggestions as Text", save_as_text(text), "suggestions.txt")
        st.download_button("Download Suggestions as Word Document", save_as_word(text), "suggestions.docx")

if __name__ == "__main__":
    main()
