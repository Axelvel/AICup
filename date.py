from os import listdir

FIRST_DATASET_PATH = 'dataset/validation_dataset/Validation_Release/'

def remove_side_char(item):
    while item[0].isdigit() == False:
        item = item[1:]
    while item[-1].isdigit() == False:
        item = item[:-1]
    return(item)

def is_possible_date(current_item,digit_number):
    count = 0
    possible = ""
    for char in current_item:
        if char == "/" or char == ".":
            possible+= "-"
            count+=1
        elif char.isdigit():
            possible+=char

    if count==2:
        return(possible)

    elif digit_number == 4 or count >2:
        possible = remove_side_char(current_item)
        
        if count >2:
            return(possible)
        if len(possible)==4 and int(possible)<2100 and int(possible)>1900 :
            return("00-00-"+possible)
    
    return False

def collect_date(path):
    list_file = listdir(path)
    file_date = []
    raw_date_strings = []

    for file in list_file:
        date = []
        raw_strings = []
        with open(path + file) as f:
            lines = f.readlines()

        for line in lines:
            if line != "\n" and sum(char.isdigit() for char in line) >= 4:
                splitted = line.split()
                for i in range(len(splitted)):
                    digit_number = sum(char.isdigit() for char in splitted[i])
                    if splitted[i].isdigit():
                        if len(splitted[i]) == 4 and 1900 < int(splitted[i]) < 2100:
                            date.append("00-00-" + splitted[i])
                            raw_strings.append(splitted[i])

                    if ("/" in splitted[i] or "." in splitted[i]) and 3 < digit_number < 9:
                        is_possible = is_possible_date(splitted[i], digit_number)
                        if is_possible:
                            date.append(is_possible)
                            raw_strings.append(splitted[i])

        if date:
            file_date.append(date)
            raw_date_strings.append(raw_strings)

    return file_date, raw_date_strings


def trivial_date(date, raw):
    trivial = []
    raw_trivial = []  # New list to store raw strings corresponding to trivial dates

    for file, raw_strings in zip(date, raw):
        file_trivial = []
        file_raw_trivial = []
        max_first = 0
        max_second = 0

        for item, raw_string in zip(file, raw_strings):
            splitted = item.split("-")
            if len(splitted) == 3 and splitted[0] != "" and splitted[1] != "" and splitted[2] != "":
                if int(splitted[0]) > max_first:
                    max_first = int(splitted[0])
                if int(splitted[1]) > max_second:
                    max_second = int(splitted[1])

        if max_second > 12 and max_first <= 12:
            for item, raw_string in zip(file, raw_strings):
                splitted = item.split("-")
                if len(splitted) == 3 and splitted[0] != "" and splitted[1] != "" and splitted[2] != "":
                    file_trivial.append(f"{splitted[1]}/{splitted[0]}/{splitted[2]}")
                    file_raw_trivial.append(raw_string)
            trivial.append(file_trivial)
            raw_trivial.append(file_raw_trivial)

        elif max_first > 12 and max_second <= 12:
            for item, raw_string in zip(file, raw_strings):
                splitted = item.split("-")
                if len(splitted) == 3 and splitted[0] != "" and splitted[1] != "" and splitted[2] != "":
                    file_trivial.append(f"{splitted[0]}/{splitted[1]}/{splitted[2]}")
                    file_raw_trivial.append(raw_string)
            trivial.append(file_trivial)
            raw_trivial.append(file_raw_trivial)

        else:
            for item, raw_string in zip(file, raw_strings):
                splitted = item.split("-")
                if len(splitted) == 3 and splitted[0] != "" and splitted[1] != "" and splitted[2] != "":
                    file_trivial.append(item)
                    file_raw_trivial.append(raw_string)
            trivial.append(file_trivial)
            raw_trivial.append(file_raw_trivial)

    return trivial, raw_trivial


def month_assumption(date, raw):
    total_date = []
    total_raw = []
    for file,raw_strings in zip(date, raw):
        first = []
        second = []
        file_date = []
        file_raw=[]

        for item in file:
            splitted = item.split("-")
            if len(splitted) != 3:
                continue
            first.append(splitted[0])
            second.append(splitted[1])
        
        n_max_first = 0
        n_max_second = 0

        for fnumber,snumber in zip(first,second):
            if first.count(fnumber) > n_max_first:
                n_max_first = first.count(fnumber)
            
            if second.count(snumber) > n_max_second:
                n_max_second = second.count(snumber)
        
        if n_max_first>n_max_second and n_max_first >1:
            for item, raw_string in zip(file, raw_strings):
                splitted = item.split("-")
                file_date.append(f"{splitted[1]}/{splitted[0]}/{splitted[2]}")
                file_raw.append(raw_string)


        elif n_max_first < n_max_second and n_max_second>1:
            for item, raw_string in zip(file, raw_strings):
                file_date.append(item.replace("-","/"))
                file_raw.append(raw_string)
        
        else:
            for item, raw_string in zip(file, raw_strings):
                file_date.append(item)
                file_raw.append(raw_string)
        total_date.append(file_date)
        total_raw.append(file_raw)
            
            
    return(total_date,total_raw)
            
def check_for_hour(string, starting_point, raw):

    if " at " in string[starting_point+len(raw):starting_point+len(raw)+10] and any(char.isdigit() for char in string[starting_point+len(raw)+4:starting_point+len(raw)+10]):
        raw_hour = string[starting_point+len(raw)+4:starting_point+len(raw)+10]
        hour_as_int = [int(char) for char in raw_hour if char.isdigit()]

        if len(hour_as_int) == 2:
            hour_as_int+= [0,0]
        elif len(hour_as_int) == 3:
            hour_as_int = [0]+hour_as_int

        if "p" in raw_hour:
            hour_as_int[0]+=1
            hour_as_int[1]+=2

        return (f"T{hour_as_int[0]}{hour_as_int[1]}:{hour_as_int[2]}{hour_as_int[3]}"),string[starting_point+len(raw):starting_point+len(raw)+10]
    return "",""

            
max = 2063
min = 1970
date,raw = collect_date(FIRST_DATASET_PATH)
date,raw = trivial_date(date,raw)
date,raw = month_assumption(date,raw)

with open("./date.txt","w") as f:
    dir = listdir(FIRST_DATASET_PATH)
    for number,(file,raw_file) in enumerate(zip(date,raw)):
        for item,raw_item in zip(file,raw_file):
            if "/" not in item:
                item = item.replace("-","/")
            splitted = item.split("/")
            if len(splitted[0])==1:
                splitted[0] = "0"+splitted[0]
            if len(splitted[1])==1:
                splitted[1] = "0"+splitted[1]
            if len(splitted[2])==2:
                buffer = (max//100)*100 + int(splitted[2])
                if buffer > max:
                    splitted[2] = str(buffer-100)
                else:
                    splitted[2] = str(buffer)
            item = splitted[0]+"/"+splitted[1]+"/"+splitted[2]
            
            with open(FIRST_DATASET_PATH + dir[number],"r") as g:
                checklines = g.read()
                x = checklines.rfind(raw_item)
            while x==-1:
                number+=1
                with open(FIRST_DATASET_PATH + dir[number],"r") as g:
                    checklines = g.read()
                    x = checklines.rfind(raw_item)

            splitted = item.split("/")
            if splitted[1]!="00" and splitted[0]!="00":
                item = splitted[2]+"-"+splitted[1]+"-"+splitted[0]
                
                hour,raw_hour = check_for_hour(checklines, x, raw_item)

                f.write(f"{dir[number][:-4]}    DATE    {x}   {x+len(raw_item)+len(raw_hour)}   {raw_item+raw_hour.strip()}  {item}{hour}")
                f.write("\n")