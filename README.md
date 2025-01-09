# 🔍 VectorDB Demo

A Streamlit application for document processing, vector embeddings visualization, and similarity search using FAISS vector database.

## Features

- **Document Processing**
  - PDF file upload and text extraction
  - Text input processing
  - Automatic text chunking with customizable parameters

- **Vector Database Operations**
  - Document embedding using HuggingFace's all-MiniLM-L6-v2 model
  - FAISS vector store integration
  - Similarity search with configurable results

- **Visualization & Analysis**
  - 2D PCA visualization of document vectors
  - Vector statistics and component analysis
  - Nearest neighbor analysis with distance metrics
  - Interactive data exploration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vectordb-demo.git
cd vectordb-demo
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Navigate to the application in your web browser (typically http://localhost:8501)

3. Use the sidebar to switch between different functionalities:
   - Add Documents: Upload PDFs or enter text
   - Query Database: Search for similar documents
   - View Contents: Explore database contents and visualizations

## Project Structure

```
vectordb-demo/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── static/
│   └── style.css      # Custom CSS styling
└── README.md          # Project documentation
```

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
