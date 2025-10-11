
from ast import mod
import datetime
from numpy.char import isdigit
from rapidfuzz import process, fuzz
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import json
from langchain_ollama import OllamaLLM
reference_list = None

from constants import *

def best_match(company_name, key="title", threshold=60):
    f = open(COMPANY_INFO_PATH,"r")
    data = json.load(f)
    reference_list = [data[key] for key in data.keys()]

    choices = [r[key].lower() for r in reference_list]
    matches = process.extract(company_name, choices, scorer=fuzz.token_set_ratio,
    limit=3,
    score_cutoff=60)
    results = []
    for match, score, idx in matches:
        results.append({
            "match": reference_list[idx][key],
            "score": score
        })
    return results
   
def find_company_names(llm, request):
    company_names = llm.invoke(f"Find the company name from the request asked , only the company names , nothing else : User request - {request}, Reminder : If there is no company name in the request , return 'NO COMPANY FOUND'")
    print(f"Company names found : {company_names}")

    company_names = company_names.lower()
    if 'no company found' in company_names:
        print("The request has no company in it")
        return
    
    company_names = company_names.strip().replace('[','').replace(']','')
    company_names = company_names.split(',')
    company_names = [name.strip() for name in company_names]

    for name in company_names:
        result = best_match(name)
        if len(result)==0:
            print(f"No valid names found for {name}")
            continue
        if result[0]['score'] > 90.00:
            print(f"Actual company name found {result[0]['match']}")
            write_company_name_to_database(result[0]['match'])
        else:
            ans = llm.invoke("Create a simple message asking for the user to enter an option from the given menu after this")
            for index, row in enumerate(result):
                print(f"{index+1} - {row['match']}")
            print(f"Any othr option -> Incorrect options")
            option = input("enter option number : ")
            option = int(option)
            if option-1 not in range(0, len(result)):
                print(f"No valid option is selected , hence this company name will not be used")
            else:
                print(f"Using the name : {result[option-1]['match']}")
                write_company_name_to_database(result[option-1]['match'])
        
def add_details_of_found_company_names():
    f = open(DATABASE_PATH,"r")
    data = json.load(f)
    if 'company_name' not in data:
        return

    data['company_details'] = {}
    f = open(TICKER_PATH, "r")
    ticker_info = json.load(f)
    for name in data['company_name']:
        for key in ticker_info.keys():
            row = ticker_info[key]
            if row['title'] == name:
                data['company_details'][name] = [row['cik_str'], row['ticker']]
                break
    g = open(DATABASE_PATH,"w")
    json.dump(data,g)
    return

def find_month_year(answer):
    months = ['january','february','march','april','may','june','july','august','september','october','november','december']
    words = answer.replace('-',' ').replace(',',' ').replace('.',' ').replace('_',' ').split(' ')
    words = [ word for word in words if len(word)>0]

    found_months = []
    found_years = []
    for word in words:
        if word.lower() in months:
            found_months.append(months.index(word.lower()) + 1)
            continue
        
        if len(word)>3 and word.isdigit():
            if int(word)>2000 and int(word)<=datetime.date.today().year:

                found_years.append(int(word))

    sorted(found_months)
    sorted(found_years)

    write_available_dates_to_database(found_months, found_years)
        
def find_dates(llm,request):

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
    ans = llm.invoke(prompt)
    find_month_year(ans)

