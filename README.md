# 📘 QueryTube — AI-Powered Semantic Video Search Platform

> **Search smarter, not harder.**

QueryTube is an AI-powered semantic search engine built on YouTube video transcripts. It enables intelligent video discovery through natural language queries, going beyond simple keyword matching to understand the meaning and context of your search.

## ✨ Features

- 🔍 **Semantic Search** — Find videos based on meaning, not just keywords
- 🤖 **AI Summarization** — Get instant video summaries using Gemini AI
- 📊 **Vector Database** — Efficient storage and retrieval with ChromaDB
- 🚀 **FastAPI Backend** — High-performance REST API for ingestion and search
- 🎨 **Flask Web Interface** — Professional, user-friendly web application
- 👤 **User Authentication** — Secure login and signup system
- 📤 **CSV Upload** — Bulk dataset ingestion capability
- 🎯 **Smart Embeddings** — 384-dimensional vectors using all-MiniLM-L6-v2

## 🏗️ Architecture

The platform follows a modern microservices architecture where user queries flow from the Flask frontend to the FastAPI backend, which performs semantic search against the ChromaDB vector store. Results are enhanced with AI-generated summaries using Gemini API, providing users with intelligent, context-aware video recommendations.

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **Vector Database** | ChromaDB |
| **Search Framework** | LangChain |
| **Backend API** | FastAPI |
| **Web Interface** | Flask |
| **Summarization** | Gemini 2.5 Flash / BART-large-CNN |
| **Frontend** | HTML, CSS, JavaScript |

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Gemini API key for summarization feature

### Installation Steps

**Step 1: Clone the Repository**

Clone this repository to your local machine using Git.

**Step 2: Create Virtual Environment**

Set up a Python virtual environment to isolate project dependencies. Use the appropriate commands for Windows or Linux/Mac systems.

**Step 3: Install Dependencies**

Install all required Python packages listed in the requirements.txt file using pip.

**Step 4: Configure Environment Variables**

Create a .env file in the project root and add your Gemini API key for the summarization feature.

### Data Pipeline Setup

**Phase 1: Data Preprocessing**

Run the preprocessing script to clean and prepare your dataset. This step removes special characters, normalizes text to lowercase, converts duration to seconds, and combines title with transcript for embedding generation.

**Phase 2: Generate Embeddings**

Execute the embedding generation script to create 384-dimensional vector representations using the all-MiniLM-L6-v2 model.

**Phase 3: Vector Database Ingestion**

Load the processed data into ChromaDB, storing video metadata including video_id, title, channel_title, transcript, duration_seconds, view_count, and embeddings.

### Running the Application

**Start Backend Service**

Launch the FastAPI backend server on port 8000. This provides the REST API for search and ingestion operations.

**Start Frontend Service**

In a separate terminal, launch the Flask web application on port 5500. This provides the user interface for searching, uploading, and viewing results.

## 🔌 API Endpoints

### FastAPI Backend (Port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest/json` | Upload single record to vector DB |
| POST | `/ingest/csv` | Bulk upload CSV with auto-embedding |
| POST | `/search` | Perform semantic search |
| GET | `/summarize/{video_id}` | Get AI-generated video summary |

### Endpoint Usage

**Semantic Search Endpoint**

Send a POST request to the search endpoint with your query and desired number of results (top_k). The API returns relevant videos ranked by similarity score.

**Summarization Endpoint**

Make a GET request to the summarize endpoint with a video_id parameter to receive an AI-generated summary of the video transcript.

## 💡 Core Functionality

### Semantic Search

The platform uses sentence transformers to convert text queries into vector embeddings, then performs similarity search against the ChromaDB vector database to find the most relevant videos based on semantic meaning rather than keyword matching.

### Video Summarization

When users request a summary, the system retrieves the video transcript and generates a concise summary using either Gemini 2.5 Flash API for fast, high-quality results, or falls back to a local BART-large-CNN model when needed.


## 📊 Data Processing Pipeline

### Stage 1: Data Preprocessing
Raw CSV data is cleaned and normalized. Titles and transcripts are sanitized by removing special characters and converting to lowercase. Duration is standardized to seconds format.

### Stage 2: Embedding Generation
The all-MiniLM-L6-v2 model generates 384-dimensional vector embeddings from the combined title and transcript text for each video.

### Stage 3: Vector Storage
Processed data with embeddings is stored in ChromaDB, creating an efficient vector database for similarity search operations.

### Stage 4: Semantic Search
User queries are converted to embeddings and compared against the database using cosine similarity. The top-K most relevant results are returned with metadata.

### Stage 5: AI Summarization
For selected videos, transcripts are processed through Gemini API or local BART model to generate concise, informative summaries.

## 🌐 Web Application Features

### Home Page
Clean, intuitive search interface with real-time results display including video thumbnails, titles, channel names, and similarity scores.

### User Authentication
Secure login and signup system allowing personalized experiences and access control.

### CSV Upload
Administrative interface for bulk uploading video datasets directly into the vector database with automatic embedding generation.

### Video Summaries
One-click access to AI-generated summaries for any video in the search results.


## 🎯 Use Cases

- **Content Creators** — Research trending topics and analyze competitor content
- **Students & Researchers** — Find educational videos by concept rather than exact keywords
- **Educators** — Discover teaching materials and create curated playlists
- **Marketers** — Identify relevant influencers and content opportunities
- **General Users** — More intuitive video discovery experience



## 📧 Contact

Sankavaram Rama Krishna - ramakrishnasankavaram436@gmail.com

---

⭐ If you find this project useful, please consider giving it a star!

**Built with ❤️ using AI and Open Source**
