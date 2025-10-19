"""
SEC Filings HTML Extractor

This module provides utilities for extracting HTML content from SEC filing documents.
SEC filings contain multiple document types embedded within a single submission file,
and this module helps isolate the specific document type needed for analysis.
"""


def extract_correct_html(filing_type, file_path):
    """
    Extract HTML content for a specific filing type from SEC submission file
    
    SEC submission files (full-submission.txt) contain multiple documents embedded
    within the same file, each marked with <TYPE> tags. This function locates the
    specific filing type and extracts its HTML content from between <TEXT> tags.
    
    Args:
        filing_type (str): The type of SEC filing to extract
                          Supported options: "S-1", "10-Q", "10-K"
                          - "S-1": IPO registration statement
                          - "10-Q": Quarterly report
                          - "10-K": Annual report
        file_path (str): Path to the SEC full-submission.txt file
    
    Returns:
        str: The extracted HTML content of the specified filing type
    
    Raises:
        ValueError: If the filing_type is not found in the file
        FileNotFoundError: If the file_path does not exist
    
    Example:
        >>> html = extract_correct_html("S-1", "sec-edgar-filings/COMPANY/S-1/full-submission.txt")
        >>> # html now contains the S-1 HTML document content
    
    Note:
        The function assumes the SEC filing follows the standard format:
        <TYPE>filing_type
        <SEQUENCE>...
        <TEXT>
        ... HTML content ...
        </TEXT>
    """
    # filing type can have these options "S-1","10-Q","10-K"
    
    # Read the entire SEC submission file into memory
    f = open(file_path, "r")
    content = f.read()

    # Step 1: Locate the position of the specific document type tag in the submission
    # SEC files contain multiple documents, each starting with <TYPE>document_name
    document_position = content.index(f"<TYPE>{filing_type}")
    
    # Step 2: Find where the actual HTML content starts (after the <TEXT> tag)
    # Add 1 to skip past the <TYPE> line, then search for <TEXT> from that position
    content_start_position = document_position+1+content[document_position:].index("<TEXT>")
    
    # Step 3: Find where the HTML content ends (at the </TEXT> closing tag)
    # Search for </TEXT> starting from the content start position
    content_end_position = content_start_position+content[content_start_position:].index('</TEXT>')

    # Step 4: Extract the HTML content between <TEXT> and </TEXT>
    # Add 6 to content_start_position to skip past the "<TEXT>" tag itself (6 characters)
    page_content_html = content[content_start_position+6:content_end_position]

    # Return the extracted HTML content for further processing
    return page_content_html

