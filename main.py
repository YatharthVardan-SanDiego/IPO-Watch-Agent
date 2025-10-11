from datetime import date
from typing import Any , Optional, Dict
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import tool
from langchain_ollama import OllamaLLM


from setup import *

"""
This is the main function file which is responsible, for orchestration.

This is a workflow file 
1. Identify company names (if present ?)
2. Store the found company names in a database accessible to every script. 
3. Identify the dates (if present ?)
4. Store the dates in a database accessible to every script.
5. If there is no company name then find the IPOs for the given dates

After this workflow , we will have two key items for summarisation of good performing 
and bad performing IPOs 
If there is only one company name then only summarise the IPO for the company
If there is only date range present then find all the IPOs in the date. 


And if neither the company name or the date is present then just summarise the IPOs for the previous month.
"""


def workflow1():
    request = input("Enter the request: ")
    llm = OllamaLLM(model = "mistral")

    #finding all the company names the request contains
    find_company_names(llm, request)
    add_details_of_found_company_names()

    #finding date ranges the request includes
    find_dates(llm, request)



workflow1()






