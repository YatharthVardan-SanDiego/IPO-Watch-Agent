"""
IPO Analysis Workflow Module

This module orchestrates the complete end-to-end workflow for analyzing IPOs.
It handles data acquisition (downloading SEC filings), data processing (ingestion into RAG),
and analysis (generating IPO scores using agent functions).

The workflow supports two modes:
1. New Companies Mode: Downloads, processes, and analyzes new IPO filings
2. Existing Companies Mode: Analyzes already-ingested companies from the vector store
"""

from symtable import Symbol
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
import os

from retriever import *
from sec_edgar_downloader import Downloader
from filings import *
from ipo_score import *
from utils import *

class WorkFlow:
    """
    IPO Analysis Workflow Orchestrator
    
    This class manages the complete pipeline for analyzing IPOs, from data acquisition
    to final scoring. It coordinates between multiple components including SEC filing
    downloads, RAG retrieval, and LLM-based analysis agents.
    
    Attributes:
        request (str): User's natural language query about IPOs
        llm: Language model instance for analysis
        rag (RAGRetriever): RAG retriever for document storage and retrieval
        dl (Downloader): SEC EDGAR downloader instance
        new_companies (bool): Flag indicating whether to process new companies or use existing
        dates (dict): Extracted date information from user query
        companies_info (list): Information about companies to analyze
        valid_companies (list): Companies with successfully downloaded filings
        ipo_objs (list): IPO scorer objects for each company
        ipo_scores (list): Final scores and summaries for each IPO
    """
    def __init__(self, query:str, rag:RAGRetriever, llm, flag = False):
        """
        Initialize the IPO Analysis Workflow
        
        Sets up all necessary components for the workflow including RAG retriever,
        LLM, and SEC downloader. Determines whether to process new companies or
        use existing ones based on the flag.
        
        Args:
            query (str): User's natural language query about IPOs (e.g., "Analyze IPOs from January 2024")
            rag (RAGRetriever): RAG retriever instance for document management
            llm: Language model instance for analysis tasks
            flag (bool): If True, downloads and processes new companies; 
                        If False, uses already-ingested companies (default: False)
        """
        # this flag tells the workflow to ingest newly found IPOs or not
        # Store the user's query for processing
        self.request = query
        # Store the LLM for agent-based analysis
        self.llm = llm
        # Store the RAG retriever for document storage/retrieval
        self.rag = rag
        # Load existing vector store to check for already-ingested companies
        self.rag.load_vectorstore()
        # Initialize SEC EDGAR downloader with organization name and email
        self.dl = Downloader("Baba Saheb Ekta Manch","kaka80197@gmail.com")
        # Set flag to determine if we're processing new companies or using existing ones
        self.new_companies = flag

    # ============================================================================
    # AGENT FUNCTION: Date Extraction from Natural Language Query
    # ============================================================================
    def extract_date(self):
        """
        [AGENT FUNCTION] Extract month and year information from user query using LLM
        
        Uses an LLM-based agent to parse the user's natural language query and extract
        temporal information (months and years) for filtering IPOs.
        
        Example:
            Query: "Analyze IPOs from January 2024"
            Output: {'months': ['January'], 'years': [2024]}
        
        Side Effects:
            Sets self.dates with extracted date information
        """
        # Use LLM agent to extract dates from natural language query
        dates = find_dates(self.llm, self.request)
        # Log the extracted dates for debugging
        print(f"WORKFLOW LOG -> Dates found {dates}")
        # Store extracted dates for later use in company filtering
        self.dates = dates

    # ============================================================================
    # DATA ACQUISITION: Get Companies Information
    # ============================================================================
    def get_companies_information(self):
        """
        Retrieve IPO company information based on extracted dates
        
        Fetches a list of companies that filed for IPO in the specified months and years.
        Uses the dates extracted from the user query to filter relevant IPOs.
        
        Prerequisites:
            Must call extract_date() first to populate self.dates
        
        Side Effects:
            Sets self.companies_info with list of company dictionaries containing:
            - company_name: Name of the company
            - symbol: Stock ticker symbol
            - ipo_date: Date of IPO filing
        """
        # Query for IPOs that match the extracted months and years
        companies_info = get_ipos_month_year(self.dates['months'], self.dates['years'])
        # Log the total number of companies found
        print(f"WORKFLOW LOG -> Total Companies Found {len(companies_info)}")
        # Store company information for download and analysis
        self.companies_info = companies_info

    # ============================================================================
    # DATA ACQUISITION: Download S-1 Filings from SEC EDGAR
    # ============================================================================
    def download_s1_filings(self):
        """
        Download S-1 filings for all companies from SEC EDGAR database
        
        Attempts to download S-1 registration statements for each company identified
        in the previous step. Handles failures gracefully and only retains companies
        with successfully downloaded filings.
        
        Prerequisites:
            Must call get_companies_information() first to populate self.companies_info
        
        Side Effects:
            Sets self.valid_companies with list of companies that have successfully
            downloaded S-1 filings
        
        Note:
            Downloads are limited to 1 filing per company (most recent)
            Failed downloads are logged but don't stop the process
        """
        # Initialize counter for tracking successful downloads
        total_new_downloads = 0
        # Initialize list to store companies with successful downloads
        valid_companies = []
        # Iterate through each company to download their S-1 filing
        for row in self.companies_info:
            print(f"WORKFLOW LOG -> Started download for S1 file of {row['company_name']}")
            try:
                # Attempt to download S-1 filing using company symbol (limit to 1 most recent filing)
                success = self.dl.get("S-1",row['symbol'], limit = 1)
                # Check if download was successful (0 indicates failure)
                if success == 0:
                    print(f"WORKFLOW LOG -> Unable to download s1 for company {row['company_name']}")
                    continue
                # Increment successful download counter
                total_new_downloads+=1
                # Add company to valid list only if download succeeded
                valid_companies.append(row)
            except Exception as e:
                # Log error but continue with other companies
                print(f"Error in downloading S-1 for {row['company_name']} -> {e}")

        # Store only companies with successfully downloaded filings
        self.valid_companies = valid_companies

        # Log total number of successful downloads
        print(f"WORKFLOW LOG -> Total New Filings Downloaded {total_new_downloads}")

    # ============================================================================
    # HELPER FUNCTION: Get File Path for S-1 Filing
    # ============================================================================
    def __file_path_for_s1(self, symbol):
        """
        [PRIVATE HELPER] Construct file path to downloaded S-1 filing
        
        Navigates the SEC EDGAR directory structure to locate the full-submission.txt
        file for a given company's S-1 filing.
        
        Directory structure: sec-edgar-filings/{symbol}/S-1/{accession_number}/full-submission.txt
        
        Args:
            symbol (str): Company stock ticker symbol
        
        Returns:
            str: Full path to the full-submission.txt file
        """
        # Construct path to company's S-1 filing directory
        path = os.path.join("sec-edgar-filings",symbol,'S-1')
        # List all accession numbers (filing identifiers) in the directory
        files= os.listdir(path)
        # Get the first accession number (most recent filing)
        accession_number = files[0]
        # Construct full path to the submission text file
        full_submission_path = os.path.join(path, accession_number,"full-submission.txt")
        return full_submission_path

    # ============================================================================
    # HELPER FUNCTION: Extract Text from S-1 Filing
    # ============================================================================
    def __fetch_text_for_s1(self, symbol):
        """
        [PRIVATE HELPER] Extract plain text from S-1 filing HTML
        
        Reads the downloaded S-1 filing, extracts the HTML content, and converts
        it to plain text for ingestion into the RAG system.
        
        Args:
            symbol (str): Company stock ticker symbol
        
        Returns:
            str: Plain text content of the S-1 filing
        
        Process:
            1. Get file path to the S-1 submission
            2. Extract S-1 HTML from the submission file
            3. Parse HTML and convert to plain text
        """
        # Get the file path to the downloaded S-1 submission
        file_path = self.__file_path_for_s1(symbol)
        # Extract the S-1 HTML content from the full submission file
        html_content = extract_correct_html("S-1",file_path)
        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        # Extract and return plain text from HTML
        return soup.get_text()

    # ============================================================================
    # DATA PROCESSING: Ingest S-1 Filings into RAG Vector Store
    # ============================================================================
    def ingest_s1_files_in_retriever(self):
        """
        Process and ingest S-1 filings into the RAG vector store
        
        Converts S-1 filing text into embeddings and stores them in the vector database
        for later retrieval during analysis. This step is only performed for new companies.
        
        Prerequisites:
            Must call download_s1_filings() first to populate self.valid_companies
        
        Note:
            - Only executes if self.new_companies flag is True
            - Currently processes only the first valid company ([:1] slice)
            - Each company's data is tagged with company_name and symbol for filtering
        
        Side Effects:
            Adds documents to the RAG vector store for each processed company
        """
        # Skip ingestion if we're using existing companies
        if not self.new_companies:
            return
        # Process only the first valid company (can be modified to process all)
        for row in self.valid_companies[:1]:
            # Extract plain text from S-1 filing
            s1_text = self.__fetch_text_for_s1(row['symbol'])
            print(f"WORKFLOW LOG -> Started ingesting {row['company_name']} info")
            # Chunk the text and add to vector store with company metadata
            self.rag.ingest_s1(s1_text, row['company_name'],row['symbol'])
            print(f"WORKFLOW LOG -> {row['company_name']} ingested into the RAG")

    # ============================================================================
    # AGENT SETUP: Create IPO Scoring Agent Objects
    # ============================================================================
    def generate_ipo_objects(self):
        """
        Instantiate IPO scoring agent objects for each company
        
        Creates an Ipo_Score agent instance for each company that will perform
        the financial, risk, and business analysis. The number of agents created
        depends on whether we're processing new companies or existing ones.
        
        Prerequisites:
            Must have self.valid_companies populated (either from download_s1_filings()
            or from self.rag.ingested_companies)
        
        Side Effects:
            Sets self.ipo_objs with list of Ipo_Score agent instances
        
        Note:
            - For new companies: Creates agents for first company only ([:1])
            - For existing companies: Creates agents for all companies
        """
        # Initialize empty list to store agent objects
        self.ipo_objs = []
        # If processing new companies, limit to first one
        if self.new_companies:
            for row in self.valid_companies[:1]:
                # Create IPO scoring agent for this company
                ipo_obj = Ipo_Score(row['company_name'], self.rag, self.llm)
                self.ipo_objs.append(ipo_obj)
        # If using existing companies, process all of them
        else:
            for row in self.valid_companies:
                # Create IPO scoring agent for this company
                ipo_obj = Ipo_Score(row['company_name'], self.rag, self.llm)
                self.ipo_objs.append(ipo_obj)
            

    # ============================================================================
    # AGENT FUNCTION: Generate IPO Scores Using Agent Analysis
    # ============================================================================
    def generate_ipo_scores(self):
        """
        [AGENT FUNCTION] Execute agent analysis and generate investment scores for all IPOs
        
        For each company, this function triggers the multi-agent analysis pipeline:
        1. Financial analysis agent
        2. Risk analysis agent
        3. Business analysis agent
        4. Final scoring agent (synthesizes all analyses)
        
        Each agent uses RAG retrieval + LLM to analyze different aspects of the IPO
        and produces a final investment recommendation.
        
        Prerequisites:
            Must call generate_ipo_objects() first to populate self.ipo_objs
        
        Side Effects:
            Sets self.ipo_scores with list of dictionaries containing:
            - company_name: Name of the company
            - score_summary: Final investment score and recommendation
        
        Note:
            This is the core agent execution step that runs all LLM-based analysis
        """
        # Initialize empty list to store final scores
        self.ipo_scores = []
        # Iterate through each IPO scoring agent object
        for obj in self.ipo_objs:
            # Execute all three analysis agents (financial, risk, business)
            obj.get_summaries()
            # Execute final scoring agent to synthesize all analyses
            score_summary = obj.generate_ipo_score()
            # Store company name and final score
            self.ipo_scores.append({'company_name':obj.company_name, 'score_summary':score_summary})

    # ============================================================================
    # MAIN ORCHESTRATION FUNCTION: Complete IPO Analysis Pipeline
    # ============================================================================
    def analyse_ipos(self):
        """
        [MAIN ORCHESTRATOR] Execute the complete end-to-end IPO analysis workflow
        
        This is the master function that orchestrates the entire pipeline from data
        acquisition to final scoring. It supports two modes of operation:
        
        Mode 1 - New Companies (self.new_companies = True):
            1. Extract dates from user query using LLM agent
            2. Get companies that filed IPOs in those dates
            3. Download S-1 filings from SEC EDGAR
            4. Ingest S-1 text into RAG vector store
            5. Create IPO scoring agents
            6. Execute multi-agent analysis and generate scores
        
        Mode 2 - Existing Companies (self.new_companies = False):
            1. Load already-ingested companies from vector store
            2. Create IPO scoring agents
            3. Execute multi-agent analysis and generate scores
        
        Returns:
            list: List of dictionaries containing IPO scores
                  Each dictionary has:
                  - 'company_name': Company name
                  - 'score_summary': Investment recommendation with rating
        
        Example:
            >>> workflow = WorkFlow("Analyze IPOs from January 2024", rag, llm, flag=True)
            >>> scores = workflow.analyse_ipos()
            >>> # scores = [{'company_name': 'Company A', 'score_summary': 'Good - Strong financials...'}]
        
        Note:
            This is the only public method that needs to be called to run the entire workflow
        """
        # Branch 1: New companies mode - full pipeline
        if self.new_companies:
            # Step 1: Extract dates from natural language query
            self.extract_date()
            # Step 2: Get list of companies that filed IPOs in extracted dates
            self.get_companies_information()
            # Step 3: Download S-1 filings from SEC EDGAR for each company
            self.download_s1_filings()
            # Step 4: Process and ingest S-1 text into RAG vector store
            self.ingest_s1_files_in_retriever()
        # Branch 2: Existing companies mode - use already ingested data
        else:
            # Load companies that are already in the vector store
            self.valid_companies = self.rag.ingested_companies 

        # Step 5: Create IPO scoring agent instances for each company
        self.generate_ipo_objects()
        # Step 6: Execute multi-agent analysis pipeline and generate final scores
        self.generate_ipo_scores()

        # Return the final IPO scores with investment recommendations
        return self.ipo_scores
            


        

        


    






            


            






    
    
    
    
