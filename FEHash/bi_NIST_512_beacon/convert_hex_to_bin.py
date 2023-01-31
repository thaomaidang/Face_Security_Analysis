import requests
import copy

def hextobin(hexval):
    thelen = len(hexval)*4
    binval = bin(int(hexval, 16))[2:]
    while ((len(binval)) < thelen):
        binval = '0' + binval
    return binval

def read_hex_code(path):
    f = open(path, 'r')
    lines = f.readlines()
    codes = []
    for i in range(len(lines)):
        tmp = []
        tmp += lines[i].strip().split("\n")
        tmp = str(tmp[0])
        codes.append(tmp)
    f.close()

    return codes

path = r'hex_NIST_1_108000.txt'
hex_codes = read_hex_code(path)
print(len(hex_codes))
bi = []
for i in range(len(hex_codes)):
    print(i)
    extract = copy.deepcopy(hex_codes[i])
    binary = hextobin(extract)

    bi.append(binary)


text_file = open("bi_NIST_1_10800.txt", "w")
for i in range(len(bi)):
    print('%s' % bi[i], file=text_file)
text_file.close()
