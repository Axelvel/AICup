from os import listdir

FIRST_DATASET_PATH = 'dataset/First_Phase_Release(Correction)/First_Phase_Text_Dataset/'

#TODO - add month detection

def collect_date(path):
    list_file = listdir(path)
    file_date = []
    for file in list_file:
        date=[]
        with open(path + file) as f:
            lines = f.readlines()
        for line in lines:
            if line != "\n" and sum(char.isdigit() for char in line)>=4:
                splitted = line.split()
                for i in range(len(splitted)):
                    digit_number = sum(char.isdigit() for char in splitted[i])
                    if splitted[i].isdigit():
                        if len(splitted[i])==4 and int(splitted[i])<2100 and int(splitted[i])>1900:
                            date.append("00-00-"+splitted[i])
                    if ("/" in splitted[i] or "." in splitted[i]) and digit_number > 3 and digit_number<9:
                        count = 0
                        possible = ""
                        for j,char in enumerate(splitted[i]):
                            if char == "/" or char == ".":
                                possible+= "-"
                                count+=1
                            elif char.isdigit():
                                possible+=char
                        if count==2:
                            date.append(possible)
                        elif count>2:
                            while possible[0] == "-":
                                possible = possible[1:]
                            while possible[-1] == "-":
                                possible = possible[:-1]
                            date.append(possible)
                        elif digit_number == 4:
                            possible = splitted[i]
                            while possible[0].isdigit() == False:
                                possible = possible[1:]
                            while possible[-1].isdigit() == False:
                                possible = possible[:-1]
                            if len(possible)==4 and int(possible)<2100 and int(possible)>1900:
                                date.append("00-00-"+possible)
                            
                        elif "00" not in splitted[i] and "mm" not in splitted[i] and splitted[i][-4]!= ".":
                            print(splitted[i])
                            

        if date != []:
            file_date.append(date)
                
    return file_date

test = collect_date(FIRST_DATASET_PATH)
#print(test)