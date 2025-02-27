# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import shutil

import uvicorn
from ai_recruiter import (
    collection,
    evaluate_candidates,
    extract_text_from_pdf,
    get_embedding,
    search_candidates,
)
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


app = FastAPI()

UPLOAD_FOLDER = "./resume_collection"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/health")
def health_check():
    """
    Health Check Endpoint.

    Returns:
        dict: A simple JSON response indicating that the service is running.
    """
    return {"status": "ok"}


@app.post("/upload/")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload and Process Resume.

    This endpoint accepts a PDF file upload, saves it locally, extracts text from the PDF,
    generates an embedding from the extracted text, and stores the resume data in ChromaDB.

    Args:
        file (UploadFile): The uploaded PDF file containing a resume.

    Returns:
        JSONResponse: A response indicating whether the file was successfully processed,
                      including the filename or an error message if text extraction fails.
    """
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Log file upload
    logger.info(f"Uploaded and saved: {file.filename}")

    # Extract text from uploaded resume
    extracted_text = extract_text_from_pdf(file_path)
    if not extracted_text.strip():
        logger.warning(f"No text extracted from {file.filename}. Skipping embedding.")
        return {"message": f"File uploaded but no text extracted: {file.filename}"}

    # Generate embedding
    embedding = get_embedding(extracted_text)
    logger.info(f"Generated embedding for: {file.filename} | Embedding size: {len(embedding)}")

    # Store new resume in ChromaDB
    collection.add(
        documents=[extracted_text],
        metadatas=[{"name": os.path.splitext(file.filename)[0]}],
        ids=[file.filename],
        embeddings=[embedding],
    )

    logger.info(f"Stored {file.filename} in ChromaDB.")

    return JSONResponse(content={"message": "File uploaded successfully", "filename": file.filename})


@app.post("/search_candidates/")
async def search_and_evaluate(top_k: int = 5):
    """
    Search and Evaluate Candidate Resumes.

    This endpoint uses a hardcoded job description to search for matching resumes stored in ChromaDB.
    It then evaluates the candidates using an AI-powered evaluation process and returns a ranked list of
    candidate evaluations along with a final hiring decision.

    Args:
        top_k (int, optional): The number of top candidate matches to retrieve. Defaults to 5.

    Returns:
        JSONResponse: A response containing the list of candidate evaluations and the final decision.
    """

    job_description = """
    We are seeking a highly motivated and experienced Software Engineer
    with strong expertise in Python and Machine Learning.
    The ideal candidate will have experience designing and developing scalable machine learning models
    and integrating them into software systems.

    Key Responsibilities:
    - Design, develop, and deploy machine learning models in production.
    - Collaborate with cross-functional teams to build scalable and efficient software systems.
    - Optimize Python code for performance and scalability.
    - Use cloud platforms like AWS, GCP, or Azure for deployment and orchestration.
    - Follow Agile methodologies and work in a collaborative environment.

    Required Skills:
    - Strong proficiency in Python and experience with machine learning libraries like TensorFlow or PyTorch.
    - Familiarity with cloud platforms (AWS, GCP, or Azure).
    - Experience with Docker, Kubernetes, and CI/CD pipelines.
    - Strong problem-solving skills and the ability to debug complex issues.
    - A degree in Computer Science, Data Science, or a related field.

    Preferred Qualifications:
    - Prior experience with NLP, computer vision, or recommendation systems.
    - Familiarity with React or Node.js is a plus.
    - Strong communication skills and experience working in Agile teams.
    """

    top_matches = search_candidates(job_description, k=top_k)
    if not top_matches:
        return JSONResponse(content={"message": "No candidates found."})

    candidate_evaluations, final_decision = evaluate_candidates(job_description, top_matches)
    return JSONResponse(content={"top_candidates": candidate_evaluations, "final_decision": final_decision})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
