import streamlit as st
import PyPDF2
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Configure page and load CSS
st.set_page_config(page_title="VectorDB Demo", layout="wide")
load_css("static/style.css")

def get_pdf_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def initialize_vector_store():
    if 'vectorstore' not in st.session_state:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_texts(["Initialize"], embeddings)
        st.session_state.document_chunks = []
        st.session_state.embeddings_model = embeddings

def add_to_vectorstore(chunks):
    embeddings = st.session_state.embeddings_model
    vectors = embeddings.embed_documents(chunks)
    st.session_state.vectorstore.add_texts(chunks)
    st.session_state.document_chunks.extend(chunks)
    if 'vectors' not in st.session_state:
        st.session_state.vectors = []
    st.session_state.vectors.extend(vectors)

def visualize_vectors(vectors):
    if len(vectors) > 1:
        vectors_array = np.array(vectors)
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors_array)
        
        df = pd.DataFrame(vectors_2d, columns=['PC1', 'PC2'])
        df['Chunk ID'] = range(len(vectors_2d))
        
        fig = px.scatter(
            df, 
            x='PC1', 
            y='PC2',
            title='2D PCA Visualization of Document Vectors',
            hover_data=['Chunk ID'],
            width=800,
            height=500
        )
        
        fig.update_traces(marker=dict(size=10))
        fig.update_layout(
            title_x=0.5,
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
        )
        return fig
    return None

def get_nearest_neighbors(vectors, index, k=5):
    vectors_array = np.array(vectors)
    distances = np.linalg.norm(vectors_array - vectors_array[index].reshape(1, -1), axis=1)
    nearest_indices = np.argsort(distances)[1:k+1]
    return nearest_indices, distances[nearest_indices]

# Main application
st.title("🔍 VectorDB Demo")
initialize_vector_store()

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h3>Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio("Choose a page", ["Add Documents", "Query Database", "View Contents"])
    
    st.markdown("""
    <div style='padding: 1rem; margin-top: 2rem; background-color: #f8f9fa; border-radius: 5px;'>
        <h4>About</h4>
        <p>This demo shows:</p>
        <ul>
            <li>Document ingestion (PDF/Text)</li>
            <li>Vector similarity search</li>
            <li>Vector space visualization</li>
            <li>Nearest neighbor analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if page == "Add Documents":
    st.header("Add Documents")
    
    upload_option = st.radio("Choose input type:", ["PDF", "Text"])
    
    if upload_option == "PDF":
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
        if uploaded_file and st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                text = get_pdf_text(uploaded_file)
                chunks = process_text(text)
                add_to_vectorstore(chunks)
                st.success(f"Added {len(chunks)} chunks to the vector database!")
            
    else:
        text_input = st.text_area("Enter your text:")
        if text_input and st.button("Process Text"):
            with st.spinner("Processing text..."):
                chunks = process_text(text_input)
                add_to_vectorstore(chunks)
                st.success(f"Added {len(chunks)} chunks to the vector database!")

elif page == "Query Database":
    st.header("Query Database")
    
    query = st.text_input("Enter your query:")
    k_results = st.slider("Number of results", min_value=1, max_value=10, value=3)
    
    if query and st.button("Search"):
        with st.spinner("Searching..."):
            results = st.session_state.vectorstore.similarity_search(query, k=k_results)
            
            query_vector = st.session_state.embeddings_model.embed_query(query)
            
            for i, doc in enumerate(results):
                with st.expander(f"Result {i+1}"):
                    st.markdown(f"""
                    <div class="result-content">
                        {doc.page_content}
                    </div>
                    """, unsafe_allow_html=True)
                    st.write("---")
                    similarity = st.session_state.vectorstore.similarity_search_with_score(query, k=1)[0][1]
                    st.write(f"Similarity Score: {similarity:.4f}")

else:  # View Contents
    st.header("Database Contents")
    
    if st.session_state.document_chunks:
        tab1, tab2, tab3 = st.tabs(["Text View", "Vector View", "Neighbor Analysis"])
        
        with tab1:
            df = pd.DataFrame({
                'Chunk ID': range(len(st.session_state.document_chunks)),
                'Content': st.session_state.document_chunks,
                'Length': [len(chunk) for chunk in st.session_state.document_chunks]
            })
            st.dataframe(df)
        
        with tab2:
            if 'vectors' in st.session_state and st.session_state.vectors:
                vectors_array = np.array(st.session_state.vectors)
                
                st.subheader("Vector Statistics")
                stats = pd.DataFrame({
                    'Statistic': ['Dimension', 'Mean', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        vectors_array.shape[1],
                        np.mean(vectors_array).round(4),
                        np.std(vectors_array).round(4),
                        np.min(vectors_array).round(4),
                        np.max(vectors_array).round(4)
                    ]
                })
                st.dataframe(stats)
                
                st.subheader("Vector Space Visualization (2D PCA)")
                try:
                    fig = visualize_vectors(vectors_array)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Need at least 2 vectors for visualization.")
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
                
                st.subheader("Sample Vector Components")
                if st.checkbox("Show raw vectors"):
                    sample_idx = st.selectbox("Select chunk to view vector:", range(len(vectors_array)))
                    st.write(f"First 10 components of vector {sample_idx}:")
                    st.code(vectors_array[sample_idx][:10])
        
        with tab3:
            if 'vectors' in st.session_state and len(st.session_state.vectors) > 1:
                st.subheader("Nearest Neighbors Analysis")
                chunk_id = st.selectbox(
                    "Select a chunk to find its nearest neighbors:",
                    range(len(st.session_state.document_chunks))
                )
                k_neighbors = st.slider("Number of neighbors", 2, 10, 5)
                
                if st.button("Find Nearest Neighbors"):
                    indices, distances = get_nearest_neighbors(
                        st.session_state.vectors,
                        chunk_id,
                        k_neighbors
                    )
                    
                    for idx, dist in zip(indices, distances):
                        with st.expander(f"Neighbor (Distance: {dist:.4f})"):
                            st.markdown(f"""
                            <div class="result-content">
                                {st.session_state.document_chunks[idx]}
                            </div>
                            """, unsafe_allow_html=True)
    else:
        st.info("No documents have been added to the database yet.")
