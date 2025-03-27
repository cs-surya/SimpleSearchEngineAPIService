

---

#  Semantic Research Paper Search API

A FastAPI-based service that fetches research papers from the **arXiv API**, encodes them using **SciBERT** (from the `sentence-transformers` library), and performs **semantic search** using **Annoy** for fast approximate nearest neighbor queries.

## Acknowledgements
Special thanks to arXiv for providing open access to research data via their public API. This project makes use of arXiv’s open access interoperability to promote accessible scientific discovery.

## Features

- 🔗 Pulls latest research papers via the arXiv API
- 🤖 Embeds abstracts using `allenai/scibert_scivocab_uncased`
- ⚡ Semantic search with `AnnoyIndex`
- 🧠 Smart matching based on abstract similarity
- 🛠️ Built with **FastAPI** for speed and developer friendliness

---

## Tech Stack

- **FastAPI** - Web framework
- **SciBERT** - Sentence embeddings for scientific language
- **Annoy** - Approximate nearest neighbor search
- **arXiv API** - Source of research papers
- **NumPy**, **Requests**, **XML** - Support libs

---

## Installation

```bash
git clone https://github.com/yourusername/semantic-research-search.git
cd semantic-research-search
pip install -r requirements.txt
```

Make sure to include this in your `requirements.txt`:

```
fastapi
uvicorn
requests
sentence-transformers
annoy
numpy
```

---

##  Running the API

```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` to explore the interactive Swagger UI.

---

## 🛠️ API Endpoints

### `GET /`
**Purpose:** Health check  
**Returns:** `{"Message": "Localhost works!"}`

---

### `GET /FetchPapersFromARXIv?query=<your_topic>`
**Description:** Fetches top arXiv papers for a given query  
**Example:**  
```bash
curl http://localhost:8000/FetchPapersFromARXIv?query=deep+learning
```

---

### `GET /SearchResearchPapers?query=<your_topic>`
**Description:** Fetches, embeds, and searches research papers semantically  
**Example:**  
```bash
curl http://localhost:8000/SearchResearchPapers?query=neural+networks
```

**Returns:** Top 3 semantically matched research papers with:
- Title
- Abstract
- First Author
- Link to paper

---

## ⚙️ How It Works

1. Sends your query to the arXiv API.
2. Extracts metadata + abstract from top results.
3. Generates embeddings using SciBERT.
4. Stores them in an `AnnoyIndex` for fast vector similarity lookup.
5. Encodes the query and returns the closest papers by semantic meaning.

---

## Future Enhancements

- Save Annoy index to disk for persistence
- Caching repeated queries
- Ranking by citation count or date
- UI frontend for interactive search
- Add user accounts and search history

---

##  Authors

Built by Surya Chandiramouli Subhashini  
Inspired by the need to **simplify academic search with ML**.

---
