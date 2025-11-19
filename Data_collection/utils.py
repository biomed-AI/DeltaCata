import requests
from bs4 import BeautifulSoup
import pubchempy as pcp
import re

# 01Download and preprocess BRENDA data.ipynb: step2
# Used to get SMILES
def get_comp(comp):
    try:
        compounds = pcp.get_compounds(comp, 'name')
        return compounds[0] if compounds else -1
    except Exception:
        return -1


# 01Download and preprocess BRENDA data.ipynb: step2
# Clean comments by normalizing stray Windows-1252 characters
def sanitize_column_text(dataset):
    for i,row in enumerate(dataset):
        if "\x96" in row[6]:
            dataset[i][6] = row[6].replace('\x96', '-')
        if "\x92" in row[6]:
            dataset[i][6] = row[6].replace('\x92', "'")
    return dataset



# 01Download and preprocess BRENDA data.ipynb: step3
# Used to extract pH and temperature information from comments
def extract_temperature_and_pH(text):
    # Matches temperature, supports °C or ºC with spaces and Â characters
    temp_pattern = r"(-?\d+(?:\.\d+)?)\s*[Â]?[°º]C?"
    ph_pattern = r"pH\s*([-+]?\d+(?:\.\d+)?)"
    temperatures = sorted(list(re.findall(temp_pattern, text)))
    ph_values = sorted(list(re.findall(ph_pattern, text)))
    return temperatures, ph_values

def Add_temperature_pH_fieds(dataset):
    dataset_with_ph_temp = []
    for row in dataset:
        temps,pHs = extract_temperature_and_pH(row[6])
        temps = '-' if len(temps)==0 else temps
        pHs = '-' if len(pHs)==0 else pHs
        temprow = row+[','.join(temps),','.join(pHs)]
        dataset_with_ph_temp.append(temprow)
    return dataset_with_ph_temp


# Used to extract mutant information from comments
amino_acid_dict = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
}

def extract_mutations(text):  
    mutation_pattern = r"\b([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})\b"
    mutations = re.findall(mutation_pattern, text)
    formatted_mutations = []
    for m in mutations:
        start_aa = amino_acid_dict.get(m[0].capitalize(),m[0])
        end_aa = amino_acid_dict.get(m[2].capitalize(),m[2])
        if len(start_aa)>1 or len(end_aa)>1:
            continue
        formatted_mutations.append(''.join([start_aa,m[1],end_aa]))
    return formatted_mutations




# 02Download and preprocess SABIO-RK data.ipynb: step2 (a)
# Used to handle entries with multiple kcat values and at least one km value
def get_page_from_sabiork(entry_id):
    # Send a request to get the web page content
    entry_data = []
    url = f"https://sabiork.h-its.org/kindatadirectiframe.jsp?kinlawid={entry_id}&newinterface=true"
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        parameter_row = soup.find('td', string='Parameter')
        if parameter_row:
            # Get the entire table
            table = parameter_row.find_parent('table')
            
            # Parsing table contents
            rows = table.find_all('tr',recursive=True)
            for row in rows[2:]:
                cols = row.find_all('td')
                data = [col.text.strip() for col in cols]
                if data[1] in ['kcat','Km']:
                    entry_data.append(data)

        else:
            print("No table with 'Parameter' header found")
    else:
        print(f"Unable to access the page, status code:{response.status_code}")
    return entry_data


# 02Download and preprocess SABIO-RK data.ipynb: step2 (b)
# Used to obtain molecular weight for the subsequent unit conversion of km
def get_mol_weight(comp):
    try:
        compound = pcp.get_compounds(comp, 'name')
    except:
        return -1
    if compound:
        return compound[0].molecular_weight
    else:
        return -1