from os import listdir

with open("./date.txt","r") as f:
    date_lines = f.readlines()
with open("./answer.txt","r") as f:
    answer_lines = f.readlines()

answer_number = 0
answer_split = answer_lines[answer_number].split()
dir = listdir("dataset/validation_dataset/Validation_Release/")

with open("combined.txt","w") as w:
    for date_line in date_lines:
        date_split = date_line.split()
        
        while answer_number < len(answer_lines)-1 and (dir.index(date_split[0]+".txt")>dir.index(answer_split[0]+".txt")):
            w.write(answer_lines[answer_number])
            answer_number+=1
            answer_split = answer_lines[answer_number].split()
        
        if(dir.index(date_split[0]+".txt")<dir.index(answer_split[0]+".txt")):
            w.write(date_line)

        else:
            while  answer_number < len(answer_lines)-1 and (int(answer_split[2])<int(date_split[2])):
                w.write(answer_lines[answer_number])
                answer_number+=1
                answer_split = answer_lines[answer_number].split()

            w.write(date_line)
            if (answer_split[2]==date_split[2]):
                answer_number+=1
