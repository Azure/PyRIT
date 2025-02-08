import os
import re
import logging
from pypdf import PdfReader
from openai import OpenAI
import pandas as pd
import chromadb
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Step 1: Initialize Chroma Client and Create Collection
# -------------------------

# Initialize Chroma client
chroma_client = chromadb.Client()

# Create or get an existing collection
collection_name = "resume_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)

# -------------------------
# Step 2: Extract Text from PDFs
# -------------------------

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            extracted = page.extract_text()
            if extracted:
                text += extracted.strip() + " "  # Ensure clean whitespace
    text = re.sub(r'\s+', ' ', text)  # Remove excessive spaces and newlines
    return text.strip()

pdf_directory = r'./resume_collection'
resumes = []

for filename in os.listdir(pdf_directory):
    if filename.lower().endswith('.pdf'):
        pdf_path = os.path.join(pdf_directory, filename)
        try:
            extracted_text = extract_text_from_pdf(pdf_path)
            resumes.append({
                'id': str(len(resumes) + 1),  # Chroma requires string IDs
                'name': os.path.splitext(filename)[0],  # Assuming filename is the candidate's name
                'text': extracted_text
            })
            logger.info(f"Processed resume: {filename}")
        # Capture pypdf EmptyFileError
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
        
# If there are no resumes, throw an error
# and inform the user of the issue
if not resumes:
    logger.error("No resume content found. Add PDFs to the resume_collection directory.")
    raise SystemExit

# -------------------------
# Step 3: Generate Embeddings
# -------------------------

client = OpenAI(api_key=os.getenv('OPENAI_KEY'))  

def get_embedding(text, model="text-embedding-3-small"):
    """Generates an embedding for the given text using OpenAI's API."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Generate embeddings for each résumé
for resume in resumes:
    resume['embedding'] = get_embedding(resume['text'])
    logger.info(f"Generated embedding for: {resume['name']} | Embedding size: {len(resume['embedding']) if resume['embedding'] else 'FAILED'}")


# -------------------------
# Step 4: Store Embeddings in ChromaDB
# -------------------------

# Create a DataFrame for easier manipulation 
df = pd.DataFrame(resumes)

# Prepare data for ChromaDB
documents = df['text'].tolist()
metadatas = df[['name']].to_dict(orient='records')  
ids = df['id'].tolist()
embeddings = df['embedding'].tolist()

# Add documents to the ChromaDB collection
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
    embeddings=embeddings
)

logger.info(f"Stored {len(resumes)} resumes in ChromaDB.")

# -------------------------
# Step 5: Perform Semantic Search with ChromaDB
# -------------------------

def search_candidates(job_description_text, k=5):
    """Searches for the top k candidates that best match the job description."""
    # Generate embedding for the job description
    job_embedding = get_embedding(job_description_text)

    # Log job description being processed
    logger.info("Performing similarity search for job description...")

    # Perform similarity search in ChromaDB
    results = collection.query(
        query_embeddings=[job_embedding],
        n_results=k,
        include=['documents', 'metadatas', 'distances']  # Ensure documents are included
    )


    if not results or not results.get('documents') or len(results['documents'][0]) == 0:
        logger.warning("No matching candidates found.")
        return []

    documents = results.get('documents', [[]])[0] or ["No content available"]
    metadatas = results.get('metadatas', [[]])[0]
    distances = results.get('distances', [[]])[0]

    top_candidates = []
    for i in range(min(len(documents), k)):  # Ensure we don't exceed available results
        result = documents[i]
        metadata = metadatas[i]
        distance = distances[i]

        logger.info(f"Candidate: {metadata.get('name', 'Unknown')} | Distance: {distance}")
        
        top_candidates.append({
            'name': metadata.get('name', 'Unknown'),
            'text': result,  # Store full text for AI Recruiter
            'distance': distance
        })

    return top_candidates

# -------------------------
# Step 6: AI Recruiter (GPT-4o) to Evaluate Résumés
# -------------------------

def evaluate_candidate(job_description, candidate_name, candidate_text, model="gpt-4o"):
    """
    Uses GPT-4o to evaluate how well a candidate matches the job description.
    Returns an evaluation summary, strengths, and a rating out of 10.
    """
    # Define system prompt with a focus on decision-making
    system_prompt = (
        "You are an experienced hiring manager tasked with evaluating candidates for a software engineering position. "
        "Each candidate has a résumé, and you are provided with the job description. For each candidate, provide:\n"
        "1. Relevant skills related to Python, machine learning, and software engineering.\n"
        "2. Specific accomplishments demonstrating experience in these areas.\n"
        "3. Gaps in their qualifications for the role.\n"
        "4. A match score from 1 to 10, with 10 being the most ideal match.\n"
        "Focus on clarity and actionable insights to help make a final decision."
    )

    # Build the conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": (
                f"Job Description:\n{job_description}\n\n"
                f"Candidate Name: {candidate_name}\n"
                f"Candidate Résumé:\n{candidate_text}"
            )
        }
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        # The GPT-4o response
        response = completion.choices[0].message.content
    except Exception as e:
        response = f"Error in generating evaluation: {str(e)}"

    return response

def evaluate_candidates(job_description, candidates, model="gpt-4o"):
    """
    Evaluates a list of candidate dictionaries (each with 'name' and 'text') 
    against a job description using GPT-4o.
    """
    evaluations = []  # To store detailed evaluations
    memory = []  # Short-term memory for decision-making

    for candidate in candidates:
        candidate_name = candidate['name']
        candidate_text = candidate['text']

        logger.info(f"Evaluating Candidate: {candidate_name}")

        # Evaluate the candidate with GPT-4o
        evaluation = evaluate_candidate(job_description, candidate_name, candidate_text, model=model)

         # Extract match score from evaluation
        match_score = extract_match_score(evaluation) or 0  # Default to 0 if not found

        logger.info(f"Match Score for {candidate_name}: {match_score}/10")

        # Append detailed evaluation for output
        evaluations.append({
            'name': candidate_name,
            'match_score': match_score,
            'distance': round(candidate['distance'], 4)
        })

        # Add to memory for decision-making
        memory.append({
            'name': candidate_name,
            'score': match_score,
            'distance': candidate['distance'],
            'text': candidate_text
        })

    # Make the final decision
    decision = make_final_decision(memory)
    logger.info(f"Final Decision: {decision}")

    return evaluations, decision

def extract_match_score(evaluation_text):
    """
    Extracts the match score from GPT-4o's evaluation text.
    Assumes the score is mentioned explicitly as "Match Score: X/10".
    """
    match = re.search(r'Match Score: (\d+)/10', evaluation_text)
    return int(match.group(1)) if match else None

def make_final_decision(memory):
    """
    Analyzes short-term memory of candidate evaluations and determines the best match.
    Uses tie-breaking logic for candidates with equal scores.
    """
    if not memory:
        return "No candidates available for decision-making."
    
    # Sort candidates by match score, then by semantic distance (ascending)
    sorted_memory = sorted(
        memory,
        key=lambda x: (x['score'], -x.get('distance', float('inf'))),  # -distance for ascending order
        reverse=True
    )
    best_candidate = sorted_memory[0]

    return (
        f"Best Candidate: {best_candidate['name']} with a Match Score of {best_candidate['score']}/10.\n"
    )