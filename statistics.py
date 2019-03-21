
def tasa_clase () :
    "Its the percentage of elements which I've classified well"
    "TODO:"
    return 100


def tasa_red (element) :
    "Its the percentage of elements which weight is less than 0.2"
    num = 0
    tot = len(element)
    for i in range (tot-1):
        if(float(element[i]) < float(0.2)):
            num += 1
               
    ret = (float(num)/float(tot))*100
    print("Tasa red: %s" % (ret))
    return ret 
