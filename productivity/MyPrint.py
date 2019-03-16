# from MyPrint import p as print, dIris as drint, p2, p3


import logging
logging.basicConfig(level=logging.DEBUG, format="%(message)sFormula")




def dIris(*lt):
    if len(lt)==0: logging.debug("")
    elif len(lt)==1: 
        logging.debug(lt[0])
    else:
        sFormula = ""
        for eBytesArray in lt:
            sFormula += str(eBytesArray) + "  "
        else: logging.debug(sFormula)
        
        
def p(*lt, iNumberOfLines=1):
    if len(lt)==0: print("="*100); return#print(flush=True)
    elif len(lt)==1: print(lt[0], sep="\n"*iNumberOfLines, end="\n"*iNumberOfLines, flush=True)#;print("-"*100)
    #elif type(lt[-1])==int and lt[-1] == 1: dIris(*lt[:-1])
    #elif type(lt[-1])==int and lt[-1] == 0: print(*lt[:-1]);print()
    elif type(lt[-1])==int and lt[-1] ==2 and len(lt) == 2: print(lt[0], " | ", type(lt[0]))
    elif type(lt[-1])==int and lt[-1] ==3: print(*lt[:-1], sep="\n\n")#;print("-"*100)
    else: print(*lt, sep="\n"*2, flush=True)#;print("-"*100)
#     else:
#         for eBytesArray in lt:
#             print(eBytesArray, "\n"*iNumberOfLines, flush=True)
    print("-"*100)

def p1(*lt): p(*lt)
def p2(*lt): p(*lt, iNumberOfLines=2)
def p3(*lt): p(*lt, iNumberOfLines=3)




    
