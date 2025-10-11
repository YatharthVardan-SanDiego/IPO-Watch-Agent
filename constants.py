from calendar import month
import json
DATABASE_PATH = "database.json"
TICKER_PATH = "company_tickers.json"

def write_company_name_to_database(company_name):
    f = open(DATABASE_PATH,"r")
    current_data = json.load(f)

    if 'company_name' not in current_data:
        current_data['company_name'] = [company_name]
    else:
        current_data['company_name'].append(company_name)

    j = open(DATABASE_PATH,"w")   
    json.dump(current_data, j)


def write_available_dates_to_database(months, years):
    f = open(DATABASE_PATH,"r")
    current_data = json.load(f)

    current_data['date_response'] = {
        'month':months,
        'years':years
    }

    j = open(DATABASE_PATH,"w")   
    json.dump(current_data, j)

COMPANY_INFO_PATH = "company_tickers.json"