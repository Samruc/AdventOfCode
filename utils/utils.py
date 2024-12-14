from datetime import datetime, timedelta

startTime = datetime.now()

def read(file, year, raw=False, join=False, strip=False):
    f = open("Data" + str(year) + "/" + file + ".txt", "r")
    data = []
    for line in f:
        if strip:
            line = line.strip()
        if raw:
            data += [line]
        else:
            items = [int(s) for s in line.strip().split()]
            if len(items) == 1:
                data += items
            else:
                data += [items]
    if join:
        return " ".join(data)
    else:
        return data

def mark(name, my_guess, ans, skip_and_add_time = None):
    global startTime
    if skip_and_add_time:
        delta = timedelta(seconds=skip_and_add_time)
        startTime -= delta
    elif my_guess != ans:
        print(name,  "guess did not match:", my_guess)
    print(name + ":", datetime.now() - startTime)
