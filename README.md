# SHL Assessment Recommendation System

## Overview

The SHL Assessment Recommendation System is a FastAPI-based application designed to recommend assessments based on user queries. It leverages machine learning techniques, including sentence embeddings and a FAISS index for efficient similarity search. The system also integrates with the Gemini API to enrich user queries, enhancing the relevance of the recommendations.

## Features

- **FastAPI Backend**: A robust API for handling requests and serving recommendations.
- **Query Enrichment**: Utilizes the Gemini API to enhance user queries with relevant keywords and skills.
- **Recommendation Engine**: Provides top-k recommendations based on enriched queries using a FAISS index for fast retrieval.
- **Streamlit Frontend**: A user-friendly web interface for inputting job descriptions and viewing recommendations.

## Technologies Used

- **FastAPI**: For building the API.
- **Pydantic**: For data validation and settings management.
- **Sentence Transformers**: For creating embeddings from assessment descriptions.
- **FAISS**: For efficient similarity search.
- **Streamlit**: For creating the web interface.
- **Python**: The programming language used for development.
- **dotenv**: For managing environment variables.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/shl-assessment-recommendation.git
   cd shl-assessment-recommendation
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory and add your Gemini API key:
   ```plaintext
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### Running the API

To start the FastAPI server, run the following command:
```bash
uvicorn main:app --reload
```
You can access the API documentation at `http://127.0.0.1:8000/docs`.

### Running the Streamlit App

To start the Streamlit application, run:
```bash
streamlit run streamlit_app.py
```
This will open a new tab in your web browser where you can enter job descriptions and view recommendations.

## API Endpoints

### POST /recommend

- **Description**: Get recommendations based on a user query.
- **Request Body**:
  ```json
  {
    "query": "string",
    "top_k": "integer"  // Optional, defaults to 5
  }
  ```
- **Response**:
  ```json
  {
    "original_query": "string",
    "enriched_query": "string",
    "recommendations": [
      {
        "assessment_name": "string",
        "assessment_type": "string",
        "url": "string",
        "remote_testing": "string",
        "adaptive_irt": "string",
        "duration": "string",
        "test_type": "string",
        "description": "string"
      }
    ]
  }
  ```

### GET /

- **Description**: Check if the API is running.
- **Response**:
  ```json
  {
    "message": "SHL Recommender API is running!"
  }
  ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the developers of FastAPI, Streamlit, and other libraries used in this project.
- Special thanks to the creators of the Gemini API for providing powerful query enrichment capabilities.
