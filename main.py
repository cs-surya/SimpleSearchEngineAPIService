from fastapi import FastAPI, Query
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import xml.etree.ElementTree as ET


app = FastAPI()

# Pre trained Science vocab model
model=SentenceTransformer("allenai/scibert_scivocab_uncased")

# arXiv API URL
ARXIV_API_URL = "http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results=50"

#Annoy Settings
VECTOR_DIM =768
annoy_index=AnnoyIndex(VECTOR_DIM,"angular")
paper_store={}
paper_counter=0

#Test API
@app.get("/")
def read_root():
    return {"Message":"Localhost works!"}

# Fetch Research Papers from ARXiv
@app.get("/FetchPapersFromARXIv")
def fetch_arxiv_papers(query):
    """Fetches top 10 research papers from arXiv API and correctly parses XML."""
    response = requests.get(ARXIV_API_URL.format(query))

    if response.status_code != 200:
        return []

    root = ET.fromstring(response.text)  # Parse XML response
    papers = []

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):  # Find all paper entries
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()

        # Extract first author
        author_elem = entry.find("{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name")
        author = author_elem.text.strip() if author_elem is not None else "Unknown"

        # Extract paper link
        link_elem = entry.find("{http://www.w3.org/2005/Atom}id")
        link = link_elem.text.strip() if link_elem is not None else "No Link"

        papers.append({
            "title": title,
            "abstract": abstract,
            "author": author,
            "link": link
        })

    return papers  # Returns a list of 10 papers (or however many arXiv provides)

@app.get("/SearchResearchPapers")
def search_papers(query: str = Query(..., description="Enter search query")):
    """Fetches arXiv papers, generates embeddings, and performs ML-powered search using Annoy."""
    global paper_store  # Store paper metadata globally

    papers = fetch_arxiv_papers(query)
    if not papers:
        return {"message": "No research papers found"}

    # Generate embeddings
    abstracts = [paper["abstract"] for paper in papers]
    embeddings = [model.encode(text) for text in abstracts]

    # Store embeddings in Annoy index
    start_index = len(paper_store)  # Start IDs from the last stored paper index
    for i, emb in enumerate(embeddings):
        annoy_index.add_item(start_index + i, emb)
        paper_store[start_index + i] = papers[i]  # Store metadata

    if annoy_index.get_n_items() > 0:
        annoy_index.build(10)  # Build the index once

    # Perform semantic search in Annoy
    query_embedding = model.encode(query)
    nearest_neighbors = annoy_index.get_nns_by_vector(query_embedding, 3)  # Top 3 matches

    results = [paper_store[i] for i in nearest_neighbors]
    return {"results": results}












