import requests
import copy

url = 'https://beacon.nist.gov/beacon/2.0/chain/1/pulse/'

for i in range(107932, 108001):
    print(i)
    new_url = url + str(i)
    r = requests.get(new_url)
    text = copy.deepcopy(r.text)

    extract = text.split(': ')[-1].split('  }')[0]
    extract = extract[:-1]

    text_file = open("NIST_107932_10800.txt", "a")
    text_file.write(extract)
    text_file.close()

