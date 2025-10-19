"""
IPO Scoring Module

This module provides functionality to analyze and score IPOs (Initial Public Offerings) 
based on financial information, risk factors, and business analysis using RAG (Retrieval-Augmented Generation)
and Large Language Models.
"""

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


from retriever import *

class Ipo_Score:
    """
    IPO Scoring Agent
    
    This class implements an intelligent agent system that analyzes IPO filings and generates
    investment recommendations. It uses RAG retrieval to fetch relevant information and LLM
    chains to analyze financial health, business prospects, and risk factors.
    
    Attributes:
        rag (RAGRetriever): The retriever instance for querying vector store
        company_name (str): Name of the company being analyzed
        llm: The language model used for analysis
        filter (dict): Filter dictionary to query company-specific documents
        financial_summary (str): Summary of financial analysis
        risks_summary (str): Summary of risk analysis
        business_summary (str): Summary of business analysis
    """
    def __init__(self, company_name, rag:RAGRetriever, llm):
        """
        Initialize the IPO Scoring Agent
        
        Sets up the agent with necessary components for analyzing IPO documents including
        the RAG retriever, company identifier, and language model for analysis.
        
        Args:
            company_name (str): Name of the company whose IPO is being analyzed
            rag (RAGRetriever): RAG retriever instance for document retrieval from vector store
            llm: Language model instance for generating analysis and summaries
        """
        # Store the RAG retriever for querying documents
        self.rag = rag
        # Store company name for filtering and identification
        self.company_name = company_name
        # Store the LLM for analysis tasks
        self.llm = llm
        # Create filter dictionary to retrieve only this company's documents
        self.filter = {
            'company_name':self.company_name
        }
        # Initialize summary attributes (populated later by analysis functions)
        self.financial_summary = None
        self.risks_summary = None
        self.business_summary = None
        
    # ============================================================================
    # AGENT FUNCTION: Financial Analysis
    # ============================================================================
    def analyse_financial_information(self):
        """
        [AGENT FUNCTION] Analyze and summarize the financial health of the company
        
        This agent function retrieves financial information from IPO documents using RAG,
        then uses an LLM to analyze profitability, growth trajectory, and overall financial health.
        
        Process:
            1. Query vector store for financial information (revenue, losses, projections)
            2. Aggregate retrieved documents into context
            3. Use LLM chain to analyze and rate the financial situation
        
        Returns:
            str: A structured summary containing:
                - Current profit/loss status
                - Growth analysis based on financials
                - Overall financial rating ('Good' or 'Bad')
        """
        # Define query to retrieve all financial data from IPO documents
        query = "Extract all the financial information about the company , everything about revenue, losses , projections"
        # Retrieve relevant documents from vector store filtered by company name
        documents = self.rag.query_vectorstore_with_filter(query,self.filter)
        # Combine all retrieved document contents into a single context string
        context = "\n\n".join([d.page_content for d in documents])
        # Define the prompt template for financial analysis
        prompt_template = """
            Summarise the financial situation for the company , in three points using the context- 
                1. Currently is the Company in profit 
                2. How is the growth for the company according to the financials
                3. In your understanding, rate the financial of an upcoming company as 'Good' Or 'Bad'

                Context : {context}

                Output Format: 
                1. Currently if the compay is in profit - 
                2. Growth reasoning - 
                3. Rating of financials for the company ('Good'/'Bad') - 
            """

        # Create prompt template with context variable
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        # Create LLM chain for processing
        chain = LLMChain(llm = self.llm, prompt = prompt)
        # Execute the chain with the aggregated context and return the analysis
        answer = chain.run({"context": context})
        return answer

    # ============================================================================
    # AGENT FUNCTION: Risk Analysis
    # ============================================================================
    def analyse_risks(self):
        """
        [AGENT FUNCTION] Analyze and evaluate risk factors for the IPO
        
        This agent function retrieves risk-related information from IPO documents,
        including risk factors, market concerns, and current market situation. It uses
        an LLM to prioritize risks and provide an overall risk assessment.
        
        Process:
            1. Query vector store for all risk factors and market concerns
            2. Aggregate retrieved risk documents into context
            3. Use LLM chain to analyze, prioritize, and grade the risk profile
        
        Returns:
            str: A structured risk assessment containing:
                - Highest risk/concern identified
                - Company's market understanding
                - Least significant risk/concern
                - Overall risk summarization
                - Risk grade rating ('Good' or 'Bad')
        """
        # Define query to retrieve all risk and market concern information
        query = "Extract all the risks mentioned and the current market situation described about the company , everything about risks , concerns , current market issues with the company"
        # Retrieve relevant risk documents from vector store filtered by company name
        documents = self.rag.query_vectorstore_with_filter(query, self.filter)
        # Combine all retrieved risk document contents into a single context string
        context = "\n\n".join([d.page_content for d in documents])

        # Define the prompt template for risk analysis with five-point evaluation
        prompt_template = """
            Summarise the Risks and Current market understanding for the company , in five points using the context- 
                1. Higest Risk / Concern 
                2. Market understanding by the company
                3. Least Risk / Concern
                4. Anything else to be summarise here
                5. In your understanding, rate the risks for the company - if the highest risk is a very very bad one given current global situation rate 'bad' or rate 'good'

                Context : {context}

                Output Format: 
                1. Highest Risk / Concern
                2. Market Understanding - 
                3. Least Risk / Concern - 
                4. Summarisation of risks - 
                5. Grade of the risk ('Good'/'Bad') - 
            """

        # Create prompt template with context variable
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        # Create LLM chain for risk processing
        chain = LLMChain(llm = self.llm, prompt = prompt)
        # Execute the chain with the risk context and return the analysis
        answer = chain.run({"context": context})
        return answer

    # ============================================================================
    # AGENT FUNCTION: Business Analysis
    # ============================================================================
    def analyse_business_information(self):
        """
        [AGENT FUNCTION] Analyze the company's business model and impact
        
        This agent function retrieves business-related information from IPO documents,
        including the company's operations, objectives, and market impact. It uses
        an LLM to summarize the business model and evaluate its effectiveness.
        
        Process:
            1. Query vector store for business operations and impact information
            2. Aggregate retrieved business documents into context
            3. Use LLM chain to analyze business model and impact success
        
        Returns:
            str: A structured business analysis containing:
                - What the company does (core business)
                - Impact the company is trying to make
                - Assessment of whether the impact has been successful
        """
        # Define query to retrieve all business model and operations information
        query = "Extract all the information about the company's business , what it is trying to , what it is doing , and the impact in terms of business"
        # Retrieve relevant business documents from vector store filtered by company name
        documents = self.rag.query_vectorstore_with_filter(query, self.filter)
        # Combine all retrieved business document contents into a single context string
        context = "\n\n".join([d.page_content for d in documents])

        # Define the prompt template for business analysis with three key aspects
        prompt_template = """
            Summarise the Business of this company , what it does , what it is trying to do- 
                1. What it does
                2. What impact is it trying to make
                3. Have the impact been successful 
                
                Context : {context}

                Output Format: 
                1. What it does - 
                2. Impact - 
                3. Successful Impact - 
            """

        # Create prompt template with context variable
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        # Create LLM chain for business analysis processing
        chain = LLMChain(llm = self.llm, prompt = prompt)
        # Execute the chain with the business context and return the analysis
        answer = chain.run({"context": context})
        return answer


    def get_summaries(self):
        """
        Orchestrator function to generate all three analysis summaries
        
        This function executes all three agent analysis functions (financial, risk, and business)
        in sequence and stores their results in instance attributes. These summaries are
        prerequisites for generating the final IPO score.
        
        Side Effects:
            - Sets self.financial_summary with financial analysis results
            - Sets self.risks_summary with risk analysis results
            - Sets self.business_summary with business analysis results
        """
        # Execute financial analysis agent and store summary
        self.financial_summary = self.analyse_financial_information()
        # Execute risk analysis agent and store summary
        self.risks_summary = self.analyse_risks()
        # Execute business analysis agent and store summary
        self.business_summary = self.analyse_business_information()

    # ============================================================================
    # AGENT FUNCTION: Final IPO Score Generation
    # ============================================================================
    def generate_ipo_score(self):
        """
        [AGENT FUNCTION] Generate final IPO investment recommendation
        
        This is the master agent function that synthesizes all three analyses (financial, risk,
        and business) to produce a final investment recommendation. It uses an expert scoring
        agent persona to evaluate the comprehensive picture and provide a 'Good' or 'Bad' rating.
        
        Prerequisites:
            Must call get_summaries() first to populate:
            - self.financial_summary
            - self.risks_summary
            - self.business_summary
        
        Process:
            1. Combines all three summary analyses into a single prompt
            2. Uses expert scoring agent persona via LLM chain
            3. Generates final binary investment recommendation
        
        Returns:
            str: Final IPO score with rating ('Good' or 'Bad') and brief explanation
        """
        # Define expert scoring agent prompt that synthesizes all three analyses
        prompt_template = """
            You are an expert scoring agent for the IPOs, you are provided with three key informations for an ipo
            Risks , Financials ( As filed by the company ) and the Business of the company

            risks : {risks}
            financial situation : {finance_situation}
            business : {business}

            Read both of these extracts and then return a well though indicator is the IPO 'GOOD' or 'BAD' to invest in. 

            Return Output : 
            Rating ('Good'/'Bad') - very very small explanation
            """

        # Create prompt template with all three analysis components as input variables
        prompt = PromptTemplate(template=prompt_template, input_variables=["risks","finance_situation","business"])
        # Create LLM chain for final scoring
        chain = LLMChain(llm = self.llm, prompt = prompt)
        # Execute the scoring chain with all three summaries and return final recommendation
        answer = chain.run({"risks": self.risks_summary, "finance_situation":self.financial_summary,"business":self.business_summary})
        return answer

    



