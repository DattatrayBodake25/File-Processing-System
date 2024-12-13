# File Processing System

## Overview
The **File Processing System** is a FastAPI-based application designed to streamline the process of handling PDF and text files. It provides functionality to:

- Upload and process PDF and text files.
- Extract key content such as text, headers, and tables.
- Generate visualizations like word clouds and analytics.
- Leverage AI to query the extracted content for insights.

---

## Repository Link
The source code for this project is available on GitHub: [File Processing System Repository](https://github.com/DattatrayBodake25/File-Processing-System)

---

## Features

### **File Upload**
- Supports file uploads through the `/upload/` endpoint.
- Accepts PDFs and plain text files.
- Saves uploaded files to a secure server directory.

### **Content Extraction**
- Extracts the following from uploaded files:
  - **Text**: Retrieves paragraphs and headers.
  - **Tables**: Parses tabular data into structured formats.
  - **OCR**: Uses Optical Character Recognition for scanned documents.

### **AI Querying**
- Queries the extracted content using Google Generative AI (Gemini API).
- Supports questions like:
  - Summaries of specific sections.
  - Key insights or data extractions.
- Returns structured JSON responses for easy integration.

### **Visualizations**
- Generates visual representations of extracted data:
  - **Word Cloud**: Creates a word cloud from extracted text.
  - **Analytics**: Produces bar charts based on header frequency.
  - **Table Visualization**: Displays extracted tables in a structured format.

---

## Project Structure

```
File-Processing-System/
|-- main.py              # Main FastAPI application code
|-- uploads/             # Directory to store uploaded files
|-- requirements.txt     # Dependencies required to run the application
|-- .env                 # Environment file for API keys
|-- README.md            # Project documentation
```

---

## Installation and Setup

### **Prerequisites**
- Python 3.8+
- pip (Python package manager)

### **Clone the Repository**
```bash
$ git clone https://github.com/DattatrayBodake25/File-Processing-System.git
$ cd File-Processing-System
```

### **Install Dependencies**
```bash
$ pip install -r requirements.txt
```

### **Set Environment Variables**
Create a `.env` file in the root directory and add your Gemini API key:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

### **Run the Application**
Start the FastAPI server:
```bash
$ uvicorn main:app --reload
```

### **Access the API**
The API will be available at: `http://127.0.0.1:8000`

---

## API Endpoints

### **1. Root Endpoint**
- **URL**: `/`
- **Method**: GET
- **Description**: Verifies that the API is running.
- **Response**:
```json
{
  "message": "Welcome to the File Processing System!"
}
```

### **2. File Upload**
- **URL**: `/upload/`
- **Method**: POST
- **Description**: Upload a PDF or text file.
- **Request**: Multipart form data with the file.
- **Response**:
```json
{
  "filename": "LLM Data Set.pdf",
  "type": "application/pdf",
  "message": "File uploaded successfully!"
}
```

### **3. Process File**
- **URL**: `/process/`
- **Method**: POST
- **Description**: Extracts content from the uploaded file.
- **Request**:
```json
{
  "filename": "LLM Data Set.pdf"
}
```
- **Response**:
```json
{
  "filename": "LLM Data Set.pdf",
  "message": "Content extracted successfully!",
  "content": {
    "paragraphs": "Extracted paragraphs...",
    "headers": ["Header 1", "Header 2"],
    "tables": [
      {
        "table_index": 1,
        "rows": 3,
        "columns": 2,
        "data": [["Column1", "Column2"], ["Value1", "Value2"]]
      }
    ]
  }
}
```

### **4. Query AI**
- **URL**: `/query/`
- **Method**: POST
- **Description**: Queries the extracted content using AI.
- **Request**:
```json
{
  "filename": "LLM Data Set.pdf",
  "query": "Summarize the financial section."
}
```
- **Response**:
```json
{
  "query": "Summarize the financial section.",
  "response": {
    "summary": "The revenue increased by 15% this year...",
    "key_points": {
      "revenue": "15% increase",
      "net_income": "5% decline"
    }
  }
}
```

### **5. Visualize Content**
- **URL**: `/visualize/`
- **Method**: POST
- **Description**: Generates visualizations from extracted content.
- **Request**:
```json
{
  "filename": "LLM Data Set.pdf",
  "visual_type": "wordcloud"
}
```
- **Response**:
- Returns a visual file (e.g., word cloud image).

---

## Testing the Application
- Use tools like **Postman** or **cURL** to test the API endpoints.
- Example cURL command to upload a file:
```bash
$ curl -X POST "http://127.0.0.1:8000/upload/" -F "file=@example.pdf"
```

---

## Demonstration
### **Video Walkthrough**
A detailed video walkthrough of the application is available - https://www.loom.com/share/edd4e26464204aa7b4f50f11193205b4?sid=b54631d5-5c00-46a0-baa5-d032bb4dbc46
---

## Technologies Used
- **FastAPI**: For building the API.
- **PyPDF2**: To extract text from PDFs.
- **Camelot**: To parse tabular data.
- **pytesseract**: For OCR on scanned documents.
- **Google Generative AI (Gemini)**: For querying extracted content.
- **Matplotlib & WordCloud**: For generating visualizations.

---

## Future Improvements
- Add support for more file types (e.g., DOCX, Excel).
- Enhance table extraction with advanced parsing techniques.
- Implement user authentication for secured API access.

---

## Contact
For any questions or feedback, feel free to reach out:
- **Author**: Dattatray Bodake
- **GitHub**: [DattatrayBodake25](https://github.com/DattatrayBodake25)
