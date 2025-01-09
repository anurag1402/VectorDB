# VectorDB
**A Streamlit application for document processing, vector embeddings visualization, and similarity search using FAISS vector database.
Features

Document Processing

PDF file upload and text extraction
Text input processing
Automatic text chunking with customizable parameters


Vector Database Operations

Document embedding using HuggingFace's all-MiniLM-L6-v2 model
FAISS vector store integration
Similarity search with configurable results


Visualization & Analysis

2D PCA visualization of document vectors
Vector statistics and component analysis
Nearest neighbor analysis with distance metrics
Interactive data exploration



Installation

Clone the repository:

bashCopygit clone https://github.com/yourusername/vectordb-demo.git
cd vectordb-demo

Create a virtual environment (recommended):

bashCopypython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required packages:

bashCopypip install -r requirements.txt
Usage

Start the Streamlit application:

bashCopystreamlit run app.py

Navigate to the application in your web browser (typically http://localhost:8501)
Use the sidebar to switch between different functionalities:

Add Documents: Upload PDFs or enter text
Query Database: Search for similar documents
View Contents: Explore database contents and visualizations



Project Structure
Copyvectordb-demo/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── static/
│   └── style.css      # Custom CSS styling
└── README.md          # Project documentation
Dependencies
See requirements.txt for a complete list of dependencies.**
