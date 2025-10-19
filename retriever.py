"""
RAG (Retrieval-Augmented Generation) Retriever Module

This module implements a RAG system for storing and retrieving S-1 IPO filing documents.
It uses vector embeddings (via Ollama) and Chroma vector database to enable semantic
search over company filings. The system supports document ingestion, chunk validation,
and filtered retrieval for company-specific queries.

Key Components:
- Vector embeddings using Ollama Llama2:13b
- Chroma vector database for persistent storage
- LLM-based chunk validation for quality control
- Metadata filtering for company-specific queries
- Document persistence and loading
"""

from symtable import Symbol
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM

# Initialize lightweight Mistral model for chunk validation
# This model is faster and used only for filtering irrelevant chunks
llm = OllamaLLM(model="mistral")

class RAGRetriever:
    """
    RAG Retriever for IPO Filing Documents
    
    This class manages the complete lifecycle of document storage and retrieval
    for IPO analysis. It handles text chunking, embedding generation, vector storage,
    and semantic search with metadata filtering.
    
    Attributes:
        text_splitter (RecursiveCharacterTextSplitter): Chunks documents into manageable pieces
        embeddings (OllamaEmbeddings): Generates vector embeddings using Llama2:13b
        vectorstore (Chroma): Vector database for storing and retrieving embeddings
        ingested_companies (list): List of companies whose filings have been ingested
    
    Key Features:
        - Semantic search over S-1 filings
        - Company-specific filtering
        - LLM-based chunk validation
        - Persistent storage across sessions
    """
    def __init__(self):
        """
        Initialize the RAG Retriever system
        
        Sets up all necessary components for document processing and retrieval:
        - Text splitter for chunking long documents
        - Embedding model for vector generation
        - Vector database for storage and retrieval
        - Tracking list for ingested companies
        
        Configuration:
            - Chunk size: 1500 characters (balances context and granularity)
            - Chunk overlap: 200 characters (maintains context continuity)
            - Embedding model: Llama2:13b (high-quality embeddings)
            - Vector store: Chroma with persistent storage
        """
        # Initialize text splitter for breaking documents into chunks
        # Chunk size of 1500 provides good balance between context and specificity
        # Overlap of 200 ensures no information is lost at chunk boundaries
        self.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False, # Use standard separators
        )
        # Initialize Ollama embeddings using Llama2:13b model
        # This model generates high-quality vector representations of text
        self.embeddings = OllamaEmbeddings(model="llama2:13b")
        # Initialize Chroma vector database with persistence
        # Collection stores all IPO filing embeddings with metadata
        self.vectorstore = Chroma(
            collection_name="ipo_filings",
            embedding_function=self.embeddings,
            persist_directory="./ipo_filings/"
        )
        # Track which companies have been ingested for reference
        self.ingested_companies = []

    # ============================================================================
    # AGENT FUNCTION: Chunk Validation using LLM
    # ============================================================================
    def chunk_valid(self, content):
        """
        [AGENT FUNCTION] Validate if a text chunk contains relevant IPO information using LLM
        
        Uses an LLM agent (Mistral) to determine if a text chunk contains information
        relevant to IPO analysis (business, financials, management, or risk factors).
        This filtering step improves retrieval quality by excluding irrelevant content
        like legal boilerplate, formatting artifacts, or navigation elements.
        
        Args:
            content (str): Text content of the chunk to validate
        
        Returns:
            bool: True if chunk is relevant, False otherwise
        
        Process:
            1. Extract first 500 characters as sample
            2. Construct binary classification prompt
            3. Invoke LLM to assess relevance
            4. Parse response for "yes" indicator
        
        Categories of Relevant Information:
            - Business operations and model
            - Financial data and projections
            - Management team and governance
            - Risk factors and concerns
        
        Example:
            >>> chunk = "The company reported revenue of $1.2M..."
            >>> retriever.chunk_valid(chunk)
            True
            
            >>> chunk = "Page 45 of 200..."
            >>> retriever.chunk_valid(chunk)
            False
        
        Note:
            - Uses global Mistral LLM for faster validation
            - Only processes first 500 chars for efficiency
            - Acts as quality filter during ingestion
        """
        # Use global lightweight LLM for chunk validation
        global llm
        # Construct prompt asking LLM to classify chunk relevance
        # Only uses first 500 characters for efficiency
        prompt = f"""
            Does this text contain relevant information about a company's business, financials, management, or risk factors? Answer yes or no.\n\nText:\n{content[:500]}
        """

        # Invoke LLM and check if response contains "yes"
        return "yes" in llm.invoke(prompt).lower()

    # ============================================================================
    # DATA PROCESSING: Ingest S-1 Filing into Vector Store
    # ============================================================================
    def ingest_s1(self, s1_text, company_name, stock_symbol) -> None:
        """
        Process and ingest S-1 filing text into the vector database
        
        Takes raw S-1 filing text and processes it through a complete ingestion pipeline:
        chunking, metadata enrichment, LLM-based validation, and vector storage.
        This enables semantic search and retrieval for IPO analysis.
        
        Args:
            s1_text (str): Complete text content of the S-1 filing
            company_name (str): Official company name for metadata
            stock_symbol (str): Company stock ticker symbol
        
        Returns:
            None (stores documents in vector database)
        
        Process:
            1. Chunk the S-1 text using RecursiveCharacterTextSplitter
            2. For each chunk:
               a. Create unique chunk ID
               b. Attach metadata (company name, filing type, index)
               c. Validate chunk relevance using LLM agent
               d. Add valid chunks to processing queue
            3. Batch add all validated chunks to vector store
            4. Record company in ingested_companies list
        
        Metadata Structure:
            - company_name: For filtering by company
            - filing_type: "S-1" (supports future expansion)
            - chunk_index_in_section: Position in document
            - source_id: Unique identifier for traceability
        
        Quality Control:
            Uses chunk_valid() LLM agent to filter out:
            - Legal boilerplate
            - Formatting artifacts
            - Navigation elements
            - Irrelevant content
        
        Performance:
            - Typical S-1 filing: 200-500 chunks
            - Processing time: ~30 minutes per filing
            - Only relevant chunks stored (saves storage/improves retrieval)
        
        Example:
            >>> retriever.ingest_s1(s1_text, "Example Corp", "EXAM")
            # Chunks and stores filing with metadata filtering
        
        Note:
            This is a computationally expensive operation due to:
            - Text chunking
            - LLM validation for each chunk
            - Embedding generation
            - Vector storage
        """
        # Initialize list to collect validated documents
        documents_for_db = []
        
        # Split S-1 text into chunks using configured text splitter
        chunks = self.text_splitter.create_documents([s1_text])
        # Process each chunk individually
        for i, chunk in enumerate(chunks):
            # Create unique identifier for this chunk
            chunk_id = f"{stock_symbol}-s1-chunk-{i}"

            # Attach metadata for filtering and traceability
            chunk_metadata = {
                "company_name": company_name,
                "filing_type": "S-1",
                "chunk_index_in_section": i,
                "source_id": chunk_id, # Use the chunk_id as a source_id for easy reference
            }
            # Create LangChain Document object with content and metadata
            doc = Document(
                page_content=chunk.page_content,
                metadata=chunk_metadata
            )
            # Validate chunk using LLM agent - only add if relevant
            if self.chunk_valid(chunk.page_content):
                documents_for_db.append(doc)
        
        # Batch add all validated documents to vector store
        # This generates embeddings and stores them in Chroma
        self.vectorstore.add_documents(documents_for_db)
        # Record this company as ingested for future reference
        self.ingested_companies.append({
            'company_name':company_name,
            'symbol':stock_symbol
        })

    # ============================================================================
    # RETRIEVAL FUNCTION: Query Vector Store
    # ============================================================================
    def query_vectorstore(self, query:str) -> list[Document]:
        """
        Perform semantic search across all documents in the vector store
        
        Uses cosine similarity between query embedding and document embeddings
        to find the most relevant text chunks across all ingested S-1 filings.
        
        Args:
            query (str): Natural language query or keyword search
                        Example: "What are the revenue projections?"
        
        Returns:
            list[Document]: Top 10 most relevant document chunks with metadata
        
        Note:
            - Returns documents from ALL companies (no filtering)
            - For company-specific queries, use query_vectorstore_with_filter()
            - Each Document includes page_content and metadata
        """
        # Perform similarity search returning top 10 matches
        return self.vectorstore.similarity_search(query,k=10)

    # ============================================================================
    # RETRIEVAL FUNCTION: Query Vector Store with Metadata Filtering
    # ============================================================================
    def query_vectorstore_with_filter(self, query:str, filter_criteria:dict) ->list[Document]:
        """
        Perform filtered semantic search for company-specific queries
        
        Combines semantic similarity search with metadata filtering to retrieve
        relevant documents for a specific company. This is the primary retrieval
        method used by the IPO analysis agents.
        
        Args:
            query (str): Natural language query
                        Example: "Extract financial information"
            filter_criteria (dict): Metadata filters
                        Example: {'company_name': 'Example Corp'}
        
        Returns:
            list[Document]: Top 10 most relevant document chunks for the specified company
        
        Process:
            1. Generate embedding for query
            2. Filter documents by metadata (e.g., company_name)
            3. Compute similarity within filtered set
            4. Return top 10 matches
        
        Usage in System:
            Used by ipo_score.py agents to retrieve company-specific information:
            - Financial analysis agent
            - Risk analysis agent  
            - Business analysis agent
        
        Example:
            >>> filter = {'company_name': 'Example Corp'}
            >>> docs = retriever.query_vectorstore_with_filter(
            ...     "What are the main risks?", 
            ...     filter
            ... )
            >>> # Returns only Example Corp's risk factor chunks
        
        Note:
            - Critical for multi-company analysis
            - Prevents information leakage between companies
            - k=10 provides sufficient context for LLM analysis
        """
        # Perform filtered similarity search returning top 10 matches
        return self.vectorstore.similarity_search(query, k = 10, filter = filter_criteria)

    # ============================================================================
    # PERSISTENCE: Save Vector Store and Company List
    # ============================================================================
    def save_vectorstore(self) -> None:
        """
        Persist vector store and company list to disk
        
        Saves the current state of the RAG system to enable data persistence
        across sessions. This includes both the vector embeddings (Chroma database)
        and the list of ingested companies (text file).
        
        Process:
            1. Persist Chroma vector store to ./ipo_filings/ directory
            2. Generate pipe-delimited company list
            3. Write company list to COMPANIES.TXT file
        
        Files Created/Updated:
            - ./ipo_filings/: Chroma vector database files
            - COMPANIES.TXT: List of ingested companies
        
        Format of COMPANIES.TXT:
            company_name|stock_symbol
            Example: "Example Corp|EXAM"
        
        Side Effects:
            - Writes to disk (may take several seconds for large databases)
            - Overwrites existing COMPANIES.TXT file
        
        Note:
            - Should be called after ingesting new companies
            - Enables loading in future sessions without re-processing
            - Critical for system persistence and efficiency
        
        Example:
            >>> retriever.ingest_s1(text, "Example Corp", "EXAM")
            >>> retriever.save_vectorstore()
            # Persists embeddings and company list to disk
        """
        # Persist the Chroma vector store to disk
        self.vectorstore.persist()
        # Initialize string to build company list
        information = ""
        # Format each company as pipe-delimited line
        for row in self.ingested_companies:
            line = f"{row['company_name']}|{row['symbol']}"
            information = f'{information}{line}\n'

        # Write company list to text file for easy loading
        f = open("COMPANIES.TXT","w")
        f.write(information)
        f.close()

    # ============================================================================
    # PERSISTENCE: Load Vector Store and Company List
    # ============================================================================
    def load_vectorstore(self) -> None:
        """
        Load previously saved vector store and company list from disk
        
        Restores the RAG system state from persistent storage, enabling analysis
        of previously ingested companies without re-processing their S-1 filings.
        This significantly reduces startup time and computational requirements.
        
        Process:
            1. Initialize Chroma vector store from persisted directory
            2. Read COMPANIES.TXT file
            3. Parse pipe-delimited company records
            4. Populate ingested_companies list
        
        Files Required:
            - ./ipo_filings/: Chroma vector database directory
            - COMPANIES.TXT: Company list file
        
        Side Effects:
            - Loads embeddings into memory
            - Sets self.vectorstore to loaded instance
            - Sets self.ingested_companies to loaded list
        
        Error Handling:
            - Fails if ./ipo_filings/ directory doesn't exist
            - Fails if COMPANIES.TXT is missing or malformed
            - Skips empty lines in COMPANIES.TXT
        
        Performance:
            - Loading is much faster than re-ingesting (~seconds vs ~30min/company)
            - Enables immediate query capability
        
        Example:
            >>> retriever = RAGRetriever()
            >>> retriever.load_vectorstore()
            >>> # Can now query previously ingested companies
            >>> companies = retriever.ingested_companies
            >>> print(companies)
            [{'company_name': 'Example Corp', 'symbol': 'EXAM'}, ...]
        
        Note:
            - Should be called at system startup
            - Must be called before querying if using existing data
            - Used in workflow.py Mode 2 (existing companies)
        """
        # Reconnect to persisted Chroma vector store
        self.vectorstore = Chroma(
            collection_name="ipo_filings",
            embedding_function=self.embeddings,
            persist_directory="./ipo_filings/"
        )

        # Read company list from text file
        content = ""
        try:
            f = open(f"COMPANIES.txt",'r')
            content = f.read()
        except Exception as e:
            content = ""
            print(f"No previous companies found")

        # Parse pipe-delimited company records
        lines = content.split('\n')
        ingested_companies = []
        for line in lines:
            # Skip empty lines
            if len(line)==0:
                continue
            # Split on pipe delimiter to extract company name and symbol
            info = line.split('|')
            # Add to ingested companies list
            ingested_companies.append({
                'company_name':info[0],
                'symbol':info[1]
            })

        # Update instance attribute with loaded companies
        self.ingested_companies = ingested_companies
        # Close file handle
        f.close()
            

