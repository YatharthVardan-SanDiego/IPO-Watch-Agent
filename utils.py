"""
Utility Functions for IPO Analysis System

This module provides helper functions for date extraction and IPO data retrieval.
It includes both LLM-based agent functions for natural language processing and
utility functions for data acquisition from external APIs.

Key Components:
- Date extraction from natural language using LLM agent
- IPO data fetching from NASDAQ API
- Date range processing utilities
"""

import datetime
import traceback
import requests
import time
from typing import List , Any , Dict
import json

#finding dates

# ============================================================================
# HELPER FUNCTION: Parse Months and Years from Text
# ============================================================================
def find_month_year(answer):
    """
    [HELPER FUNCTION] Extract month and year integers from text string
    
    Parses a text string to identify month names and year numbers. This function
    performs text normalization, keyword matching, and validation to extract
    temporal information from the LLM's response.
    
    Args:
        answer (str): Text string potentially containing month names and years
                     (e.g., "january 2024", "march-2023", "december, 2025")
    
    Returns:
        dict: Dictionary containing:
            - 'months': List of month integers (1-12)
            - 'years': List of year integers (2000-current year)
        
        If no months/years found, defaults to current month/year
    
    Process:
        1. Normalize text by replacing punctuation with spaces
        2. Split into words and filter empty strings
        3. Match words against month names (case-insensitive)
        4. Extract 4-digit years between 2000 and current year
        5. Default to current month/year if none found
    
    Example:
        >>> find_month_year("january 2024")
        {'months': [1], 'years': [2024]}
        
        >>> find_month_year("march-april 2023")
        {'months': [3, 4], 'years': [2023]}
    """
    # List of month names in lowercase for matching
    months = ['january','february','march','april','may','june','july','august','september','october','november','december']
    # Normalize text by replacing various punctuation with spaces
    words = answer.replace('-',' ').replace(',',' ').replace('.',' ').replace('_',' ').split(' ')
    # Filter out empty strings
    words = [ word for word in words if len(word)>0]

    # Initialize lists to store found months and years
    found_months = []
    found_years = []
    # Iterate through each word to identify months and years
    for word in words:
        # Check if word matches a month name (case-insensitive)
        if word.lower() in months:
            # Convert month name to integer (1-12) and add to list
            found_months.append(months.index(word.lower()) + 1)
            continue
        
        # Check if word is a 4-digit year
        if len(word)>3 and word.isdigit():
            # Validate year is between 2000 and current year
            if int(word)>2000 and int(word)<=datetime.date.today().year:
                found_years.append(int(word))

    # Sort months and years in ascending order
    sorted(found_months)
    sorted(found_years)

    # Default to current month if no months found
    if found_months == []:
        found_months = [datetime.date.today().month]
    # Default to current year if no years found
    if found_years == []:
        found_years = [datetime.date.today().year]
    
    # Return dictionary with extracted months and years
    return {
        'months':found_months,
        'years':found_years
    }

# ============================================================================
# AGENT FUNCTION: Extract Dates from Natural Language Query
# ============================================================================
def find_dates(llm,request):
    """
    [AGENT FUNCTION] Extract temporal information from natural language using LLM
    
    This is a key agent function that uses an LLM to parse user queries and extract
    date information (months and years). The agent is carefully prompted to handle
    various date formats and edge cases.
    
    Args:
        llm: Language model instance (e.g., OllamaLLM)
        request (str): User's natural language query
                      Examples:
                      - "Analyze IPOs from January 2024"
                      - "Show me recent IPOs"
                      - "IPOs from March to June 2023"
    
    Returns:
        dict: Dictionary containing:
            - 'months': List of month integers (1-12)
            - 'years': List of year integers
    
    Process:
        1. Construct detailed prompt with current date context
        2. Instruct LLM to extract only month/year information
        3. Handle various formats: single month, date ranges, "recent"
        4. Pass LLM response to find_month_year() for parsing
        5. Return structured date information
    
    Prompt Engineering:
        - Provides current date context
        - Specifies valid response formats
        - Instructs to avoid explanations or greetings
        - Handles edge cases (no date, "recent", ranges)
    
    Example:
        >>> find_dates(llm, "Show me IPOs from January 2024")
        {'months': [1], 'years': [2024]}
        
        >>> find_dates(llm, "Recent IPOs")
        {'months': [10], 'years': [2025]}  # current month/year
    
    Note:
        This agent function is critical for the workflow as it enables natural
        language interaction with the IPO analysis system.
    """
    # Construct detailed prompt for LLM date extraction agent
    # Provides context, instructions, and expected output format
    prompt = f"""
        You are provided with a user request , identify the date range or month or year provided in the user request ,and don't output any explanation please

        User Request : {request}

        Information : 
            Current Year - {datetime.date.today().year}
            Current Month - {datetime.date.today().month}

        Action Items : 

            1. If there is no date present in the request , return anything apart from a valid date
            2. The date range or the month can always be a past date , or a very very old date compared to current dates.
            3. The request's answer should never include a particular date only month / year / None.
            4. If there is only month specified only return month and current year
            5. If there is month and year specified return month and year
            6. If there is a range of date present in the request , make sure to return all the start month , start year and end month , end year
            7. Refrain from greeting messages , thought of chain messages , curteous messages etc , just return the answer asked
            8. If the request says recent , then return current year and current month
        Valid responses : 
            1. Nothing
            2. Month , Year
            3. Month
            4. Start month , Start year - End Month , End Year

        Output : 
            1. No other information apart from the actual answer
            2. No curteous messages 
            3. No explanation please 
            4. Simple answer that is it
        
        Reminder : 
            Please make sure , no other text or characters valid responses are just the month names / year or nothing
            Do not return any invalid date range which is not present in the request.
    """
    # Invoke LLM agent to process the query
    ans = llm.invoke(prompt)
    # Parse LLM's response to extract structured date information
    return find_month_year(ans)

# ============================================================================
# HELPER FUNCTION: Generate Date Range
# ============================================================================
def _date_range(months, years) -> list[str]:
    """
    [PRIVATE HELPER] Generate list of YYYY-MM date strings from month/year lists
    
    Creates a continuous date range from start month/year to end month/year.
    Handles single dates, same-year ranges, and multi-year ranges.
    
    Args:
        months (List[int]): List of month integers (1-12)
                           Uses first and last elements as start/end
        years (List[int]): List of year integers
                          Uses first and last elements as start/end
    
    Returns:
        List[str]: List of date strings in "YYYY-MM" format
                  Example: ['2024-01', '2024-02', '2024-03']
    
    Process:
        1. Extract start (first) and end (last) months/years
        2. If same year, ensure months are in correct order
        3. If single month/year, return single-element list
        4. Generate all months between start and end dates
        5. Handle year rollovers (December to January)
    
    Examples:
        >>> _date_range([1, 3], [2024, 2024])
        ['2024-01', '2024-02', '2024-03']
        
        >>> _date_range([12], [2023])
        ['2023-12']
        
        >>> _date_range([11, 2], [2023, 2024])
        ['2023-11', '2023-12', '2024-01', '2024-02']
    """
    # Extract start and end months from list
    month1 = months[0]
    month2 = months[-1]

    # Extract start and end years from list
    year1 = years[0]
    year2 = years[-1]

    # If same year, ensure months are in chronological order
    if year1==year2:
        month1 = min(month1, month2)
        month2 = max(month1, month2)

    # Format start date as YYYY-MM
    start_date = f"{year1}-{str(month1).zfill(2)}"

    # If single month/year, return single date
    if month1==month2 and year1 == year2:
        return [start_date]

    # Format end date as YYYY-MM
    end_date = f"{year2}-{str(month2).zfill(2)}"

    # Initialize current date to start date
    date = start_date

    # Initialize list to store all dates in range
    range= []
    # Iterate through months from start to end
    while date != end_date:
        range.append(date)

        # Handle year rollover from December to January
        if month1 == 12:
            month1 = 1
            year1 = year1+1
        else:
            # Increment month
            month1+=1
        
        # Update current date string
        date = f"{year1}-{str(month1).zfill(2)}"

    # Append end date to complete the range
    range.append(end_date)
    return range

# ============================================================================
# HELPER FUNCTION: Make API Request to NASDAQ
# ============================================================================
def _request_information(date, session):
    """
    [PRIVATE HELPER] Fetch IPO calendar data from NASDAQ API for a specific date
    
    Makes an HTTP GET request to the NASDAQ IPO calendar API to retrieve
    information about IPOs (priced, upcoming, and filed) for a given month.
    
    Args:
        date (str): Date string in "YYYY-MM" format (e.g., "2024-01")
        session (requests.Session): Requests session object with configured headers
    
    Returns:
        dict: JSON response from NASDAQ API containing:
            - 'data': IPO information
                - 'priced': Companies with priced IPOs
                - 'upcoming': Companies with upcoming IPOs
                - 'filed': Companies that have filed for IPO
    
    Raises:
        requests.HTTPError: If response status is not 200
        Exception: If unable to extract information for the date
    
    Process:
        1. Set browser-like headers to mimic legitimate request
        2. Make GET request to NASDAQ API with date parameter
        3. Validate response status
        4. Decode response content if necessary
        5. Parse and return JSON data
    
    Note:
        Headers are configured to match browser requests for better API compatibility.
        Timeout is set to 15 seconds to prevent hanging.
    """
    # Configure headers to mimic browser request for API compatibility
    headers = {
    "accept": "*/*",
    "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
    "content-type": "text/plain;charset=UTF-8",
    "priority": "u=4, i",
    "sec-ch-ua": "\"Google Chrome\";v=\"141\", \"Not?A_Brand\";v=\"8\", \"Chromium\";v=\"141\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"macOS\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "no-cors",
    "sec-fetch-site": "cross-site",
    "sec-fetch-storage-access": "active"
  }

    # Construct NASDAQ API URL with date parameter
    url = f"https://api.nasdaq.com/api/ipo/calendar?date={date}"
    # Make GET request with configured headers and 15-second timeout
    response = session.get(url,headers = headers, timeout = 15)
    # Raise exception if status code indicates error
    response.raise_for_status()

    # Validate successful response
    if response.status_code!=200:
        raise f"Unable to extract information for date {date}"
    # Get response content
    data = response.content
    # Decode bytes to string if necessary
    if not isinstance(data, str):
        data = data.decode('utf-8')

    # Parse JSON and return data
    return json.loads(data)

# ============================================================================
# DATA ACQUISITION FUNCTION: Fetch IPO Information from NASDAQ
# ============================================================================
def get_ipos_month_year(months: List[int], years: List[int]) -> List[Dict]:
    """
    Fetch comprehensive IPO information from NASDAQ API for specified date range
    
    This is the main data acquisition function that retrieves information about
    all IPOs (priced, upcoming, and filed) for a given time period by querying
    the NASDAQ IPO calendar API.
    
    Args:
        months (List[int]): List of month integers (1-12)
        years (List[int]): List of year integers
    
    Returns:
        List[Dict]: List of dictionaries, each containing:
            - 'company_name': Name of the company
            - 'ipo_date': IPO filing date (YYYY-MM format)
            - 'symbol': Proposed ticker symbol
        
        Returns dict with 'error' key if request fails
    
    Process:
        1. Generate date range using _date_range()
        2. Create requests session with browser-like headers
        3. For each date in range:
           a. Wait 0.5 seconds (rate limiting)
           b. Fetch IPO data from NASDAQ API
           c. Extract three categories: priced, upcoming, filed
           d. Parse company information from each category
           e. Aggregate all companies into list
        4. Handle errors gracefully with logging
        5. Return complete list of companies
    
    IPO Categories:
        - **Priced**: IPOs that have been priced and are trading
        - **Upcoming**: IPOs scheduled but not yet priced
        - **Filed**: Companies that have filed for IPO but not scheduled
    
    Example:
        >>> companies = get_ipos_month_year([1, 2], [2024, 2024])
        >>> print(companies[0])
        {
            'company_name': 'Example Corp',
            'ipo_date': '2024-01',
            'symbol': 'EXAM'
        }
    
    Error Handling:
        - Logs errors for individual date failures but continues processing
        - Returns error dict if entire operation fails
        - Uses traceback for detailed error information
    
    Note:
        - Includes 0.5 second delay between requests for API rate limiting
        - Used by workflow.py in the data acquisition phase
        - Critical function for identifying companies to analyze
    """
    # Generate list of dates to query (YYYY-MM format)
    date_range = _date_range(months, years)
    # Log the date range being processed
    print(f"TOOL LOG : {date_range}")
    # Initialize list to store all company information
    company_names = []
    # Create persistent session for efficient HTTP requests
    session = requests.Session()

    # Configure session with browser-like headers for API compatibility
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        # Add any other standard headers you see in Postman that are always sent
    })


    try:
        # Iterate through each date in the range
        for date in date_range:
            # Add delay to respect API rate limits
            time.sleep(0.5)
            # Fetch IPO data for this specific date
            data = _request_information(date, session)
            try:
                # Extract data payload from response
                data = data['data']
                # Extract three categories of IPO information
                priced_information = data['priced']
                upcoming_information = data['upcoming']['upcomingTable']
                filed_information = data['filed']

                # Process priced IPOs
                rows = priced_information['rows']
                if rows:
                    for row in rows:
                        # Extract and store company information
                        company_names.append({
                            'company_name':row['companyName'],
                            'ipo_date':date,
                            'symbol':row['proposedTickerSymbol']
                        })
                # Process upcoming IPOs
                rows = upcoming_information['rows']
                if rows:
                    for row in rows:
                        # Extract and store company information
                        company_names.append({
                            'company_name':row['companyName'],
                            'ipo_date':date,
                            'symbol':row['proposedTickerSymbol']
                        })

                # Process filed IPOs
                rows = filed_information['rows']
                if rows:
                    for row in rows:
                        # Extract and store company information
                        company_names.append({
                            'company_name':row['companyName'],
                            'ipo_date':date,
                            'symbol':row['proposedTickerSymbol']
                        })

            except Exception as e:
                # Log error but continue processing other dates
                print(f"Error in data extraction for the date {date} {e}")
                print(traceback.format_exc())
    except Exception as e:
        # Return error information if entire operation fails
        return {"error":f'{e}'}
    # Return complete list of companies from all dates
    return company_names



