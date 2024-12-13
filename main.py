#Importing all required libraries
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import os
from typing import Optional, List, Literal
import pytesseract
from PIL import Image
import pdf2image
import re
import PyPDF2
import camelot
import logging
from dotenv import load_dotenv
from charset_normalizer import detect
import google.generativeai as genai
from fastapi.responses import JSONResponse, FileResponse
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import pandas as pd



# Initialize environment variables and directories
def initialize_environment():
    """Load environment variables and create necessary directories."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key is missing. Please set it in the .env file.")
    os.makedirs("uploads", exist_ok=True)
    return api_key

GEMINI_API_KEY = initialize_environment()

# Initialize FastAPI app and logger
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
UPLOAD_DIR = "uploads"

# Pydantic models
class ProcessRequest(BaseModel):
    filename: str

class QueryRequest(BaseModel):
    filename: str
    query: str

class VisualizeRequest(BaseModel):
    filename: str
    visual_type: Literal["table", "wordcloud", "analytics"]


# Utility Functions
def clean_text(text: str) -> str:
    """Remove unnecessary whitespace and newline characters from text."""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
    return clean_text(text)


def extract_text_with_ocr(file_path: str) -> str:
    """Extract text from a file using OCR."""
    try:
        images = pdf2image.convert_from_path(file_path)
        text = " ".join([pytesseract.image_to_string(image) for image in images])
        return clean_text(text)
    except Exception as e:
        logger.error(f"Error using OCR: {str(e)}")
        return ""


def extract_paragraphs(text: str) -> str:
    """Extract paragraphs from text using regex."""
    paragraphs = re.findall(r'(?:(?:[A-Z][^.!?]*[.!?])\s*){2,}', text)
    return "\n\n".join(paragraphs)


def extract_headers(text: str) -> List[str]:
    """Extract headers from text using predefined patterns."""
    header_patterns = [
        r'^\d{1,2}\.\s+[A-Z][A-Za-z0-9\s\-:]{2,50}$',
        r'^[A-Z\s\-]{3,50}$'
    ]
    combined_pattern = "|".join(header_patterns)
    headers = re.findall(combined_pattern, text, re.MULTILINE)
    return [h.strip() for h in headers if len(h.split()) > 1]


def extract_tables(pdf_path: str) -> List[dict]:
    """Extract tables from a PDF file."""
    try:
        tables = camelot.read_pdf(pdf_path, pages='all')
        return [
            {
                "table_index": i + 1,
                "rows": table.df.shape[0],
                "columns": table.df.shape[1],
                "data": table.df.values.tolist()
            } for i, table in enumerate(tables)
        ]
    except Exception as e:
        logger.error(f"Error extracting tables: {str(e)}")
        return []


def extract_key_points(summary: str) -> dict:
    """Extract key points such as revenue and net income from a summary."""
    key_points = {"revenue": "N/A", "net_income": "N/A", "investment_proposals": "N/A"}
    try:
        revenue_match = re.search(r"\$(\d+(?:\.\d+)?[MB]?) million.*?total revenue", summary, re.IGNORECASE)
        if revenue_match:
            key_points["revenue"] = f"${revenue_match.group(1)}M"
        net_income_match = re.search(r"(\d+% decrease.*?profit margins|\-\$\d+[MB]?)", summary, re.IGNORECASE)
        if net_income_match:
            key_points["net_income"] = net_income_match.group(1)
        investment_matches = re.findall(r"(\$\d+(?:\.\d+)?[MB]?).*?(investment|reallocation)", summary, re.IGNORECASE)
        if investment_matches:
            key_points["investment_proposals"] = ", ".join([f"{match[0]} {match[1]}" for match in investment_matches])
    except Exception as e:
        logger.error(f"Error extracting key points: {e}")
    return key_points


def save_file(file: UploadFile) -> str:
    """Save uploaded file to the server and handle encoding issues for text files."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            content = file.file.read()
            f.write(content)
        if file.content_type == "text/plain":
            detected = detect(content)
            if detected['encoding'] and detected['encoding'].lower() != 'utf-8':
                content = content.decode(detected['encoding']).encode('utf-8')
                with open(file_path, "wb") as f:
                    f.write(content)
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    return file_path


async def extract_content(file_path: str, filename: str) -> Optional[dict]:
    """Extract paragraphs, headers, and tables from the uploaded file."""
    extracted_data = {"paragraphs": "", "headers": [], "tables": []}
    if filename.endswith(".pdf"):
        pdf_text = extract_text_from_pdf(file_path)
        if not pdf_text.strip():
            pdf_text = extract_text_with_ocr(file_path)
        extracted_data["paragraphs"] = extract_paragraphs(pdf_text)
        extracted_data["headers"] = extract_headers(pdf_text)
        extracted_data["tables"] = extract_tables(file_path)
    elif filename.endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text_content = file.read()
                extracted_data["paragraphs"] = text_content
                extracted_data["headers"] = extract_headers(text_content)
        except Exception as e:
            logger.error(f"Error reading text file: {str(e)}")
            return {"error": f"Error reading text file: {str(e)}"}
    else:
        return {"error": "Unsupported file format. Please upload a PDF or text file."}
    return extracted_data


# Endpoints
@app.get("/")
def read_root():
    """Root endpoint for the API."""
    return {"message": "Welcome to the File Processing System!"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and save it on the server."""
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and text files are allowed.")
    file_path = save_file(file)
    return {"filename": file.filename, "type": file.content_type, "message": "File uploaded successfully!"}


@app.post("/process/")
async def process_file(request: ProcessRequest):
    """Process an uploaded file to extract content."""
    file_path = os.path.join(UPLOAD_DIR, request.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found. Please upload the file first.")
    content = await extract_content(file_path, request.filename)
    if "error" in content:
        raise HTTPException(status_code=500, detail=content["error"])
    return {"filename": request.filename, "message": "Content extracted successfully!", "content": content}


@app.post("/query/")
async def query_llm(query_request: QueryRequest):
    """Query the LLM with content from the uploaded file."""
    file_path = os.path.join(UPLOAD_DIR, query_request.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found. Please upload the file first.")
    try:
        uploaded_file = genai.upload_file(file_path)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            contents=["Please provide a plain text response for the following query:", query_request.query, uploaded_file]
        )
        if not response or not response.text:
            raise HTTPException(status_code=500, detail="No response from Gemini API")
        summary = response.text.replace("\n", " ").strip()
        return {"query": query_request.query, "response": {"summary": summary, "key_points": extract_key_points(summary)}}
    except Exception as e:
        logger.error(f"Error querying Gemini API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying LLM: {str(e)}")


@app.post("/visualize/")
async def visualize_content(request: VisualizeRequest):
    """Generate visualizations for extracted content."""
    file_path = os.path.join(UPLOAD_DIR, request.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found. Please upload the file first.")
    content = await extract_content(file_path, request.filename)
    if "error" in content:
        raise HTTPException(status_code=500, detail=content["error"])

    if request.visual_type == "table":
        if not content["tables"]:
            raise HTTPException(status_code=404, detail="No tables found in the file.")
        return JSONResponse(content={"tables": content["tables"]})

    elif request.visual_type == "wordcloud":
        if not content["paragraphs"]:
            raise HTTPException(status_code=404, detail="No text content found for word cloud generation.")
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(content["paragraphs"])
        wordcloud_path = os.path.join(UPLOAD_DIR, "wordcloud.png")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(wordcloud_path, format="png")
        plt.close()
        return FileResponse(wordcloud_path, media_type="image/png", filename="wordcloud.png")
    elif request.visual_type == "analytics":
        if not content["headers"]:
            raise HTTPException(status_code=404, detail="No headers found for analytics.")
        header_counts = pd.Series(content["headers"]).value_counts()
        analytics_path = os.path.join(UPLOAD_DIR, "analytics.png")
        plt.figure(figsize=(10, 5))
        header_counts.plot(kind="bar")
        plt.title("Header Frequency")
        plt.xlabel("Headers")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(analytics_path, format="png")
        plt.close()
        return FileResponse(analytics_path, media_type="image/png", filename="analytics.png")
    else:
        raise HTTPException(status_code=400, detail="Unsupported visual_type. Use 'table', 'wordcloud', or 'analytics'.")