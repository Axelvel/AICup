from os import listdir

FIRST_DATASET_PATH = 'dataset/First_Phase_Release(Correction)/First_Phase_Text_Dataset/'

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
                        is_possible = is_possible_date(splitted[i],digit_number)
                        if is_possible != False:
                            date.append(is_possible)

        if date != []:
            file_date.append(date)
                
    return file_date

def trivial_date(date):
    trivial = []
    for file in date:
        file_trivial = []
        max_first = 0
        max_second = 0

        for item in file:
            splitted = item.split("-")
            if len(splitted)== 3 and splitted[0]!="" and splitted[1]!="" and splitted[2]!="":
                if int(splitted[0])>max_first:
                    max_first = int(splitted[0])
                if int(splitted[1])>max_second:
                    max_second = int(splitted[1])

        if max_second>12 and max_first <= 12:
            for item in file:
                splitted = item.split("-")
                if len(splitted)== 3 and splitted[0]!="" and splitted[1]!="" and splitted[2]!="":
                    file_trivial.append(f"{splitted[1]}/{splitted[0]}/{splitted[2]}")
            trivial.append(file_trivial)
        
        elif max_first>12 and max_second<=12:
            for item in file:
                splitted = item.split("-")
                if len(splitted)== 3 and splitted[0]!="" and splitted[1]!="" and splitted[2]!="":
                    file_trivial.append(f"{splitted[0]}/{splitted[1]}/{splitted[2]}")
            trivial.append(file_trivial)
        
        else:
            for item in file:
                splitted = item.split("-")
                if len(splitted)== 3 and splitted[0]!="" and splitted[1]!="" and splitted[2]!="":
                    file_trivial.append(item)
            trivial.append(file_trivial)

    return trivial

def month_assumption(date):
    total_date = []
    for file in date:
        first = []
        second = []
        file_date = []

        for item in file:
            splitted = item.split("-")
            if len(splitted) != 3:
                continue
            first.append(splitted[0])
            second.append(splitted[1])
        
        max_first = 0
        n_max_first = 0
        max_second = 0
        n_max_second = 0

        for fnumber,snumber in zip(first,second):
            if first.count(fnumber) > n_max_first:
                n_max_first = first.count(fnumber)
                max_first = fnumber
            
            if second.count(snumber) > n_max_second:
                n_max_second = second.count(snumber)
                max_second =snumber
        
        if n_max_first>n_max_second and n_max_first >1:
            for item in file:
                splitted = item.split("-")
                file_date.append(f"{splitted[1]}/{splitted[0]}/{splitted[2]}")

        elif n_max_first < n_max_second and n_max_second>1:
            for item in file:
                file_date.append(item.replace("-","/"))
        
        else:
            for item in file:
                file_date.append(item)
        total_date.append(file_date)
            
            
    return(total_date)
            
            

date = collect_date(FIRST_DATASET_PATH)
date = trivial_date(date)
date = month_assumption(date)

for file in date:
    print("file")
    for item in file:
        if "/" not in item:
            print(item)