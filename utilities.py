#!/usr/bin/python3

"""
some usefull utilities on one page
"""

__version__ = "1.0.1"


#================================================================================================

def download_ndarray(url:"url of .npy file") -> "numpy-array":
    """download a (numpy-pickled) ndarray from the web"""
    #URL = r"https://github.com/leztien/datasets/blob/master/korean.npy?raw=true"   #url example
    from numpy import load
    from urllib.request import urlopen
    from urllib.error import URLError
    from tempfile import TemporaryFile
    from shutil import copyfileobj
    from sys import exit
    
    try: rs = urlopen(URL)   # rs = response-object
    except URLError: print("unable to download"); exit()
    
    with rs as rs, TemporaryFile(mode='w+b', suffix='.npy', delete=True) as fh:
        if rs.getcode() != 200: print("unable to download"); exit()
        copyfileobj(rs, fh)  
        fh.seek(0)
        nd = load(fh)
    #just in case:
    rs.close(); fh.close(); del rs, fh
    return(nd)

#=================================================================================================

def load_from_github(url):
    """load a python module from github"""
    from urllib.request import urlopen
    from os import remove
    
    obj = urlopen(path)
    assert obj.getcode()==200,"unable to open"

    s = str(obj.read(), encoding="utf-8")
    NAME = "_temp.py"
    with open(NAME, mode='wt', encoding='utf-8') as fh: fh.write(s)
    module = __import__(NAME[:-3])
    remove(NAME)
    return module

#=================================================================================================









def main():
  pass
if __name__=='__main__':main()
