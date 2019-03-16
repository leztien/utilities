


def compare_console_output_of_two_functions(function1, function2, *args, **kwargs):  # args,kwargs must be identical for both functions
    import sys      #Note: from sys import stdout will not work because "stdout = fh" will mean assigning a value to the new variable
    from tempfile import TemporaryFile

    temp_stdout = sys.stdout    # reroute the stream temporarily

    with TemporaryFile(prefix='temp_', mode='w+t', encoding='utf-8') as fh1,\
         TemporaryFile(prefix='temp_', mode='wt+', encoding='utf_8') as fh2:
        sys.stdout = fh1    # route the output stream into the temporary file 1
        function1(*args, **kwargs)     # execute function 1 and let it send its output into the stream
        setattr(sys, 'stdout', fh2)  # same as: sys.stdout = fh2
        function2(*args, **kwargs)     # execute function 2 and let it send its output into the stream
        fh1.seek(0)     # put the cursor into the first position to read the text
        fh2.seek(0)
        s1,s2 = fh1.read(), fh2.read()
    sys.stdout = temp_stdout # restore the output channel to its original (i.e. the console)
    sys.__setattr__('stdout', temp_stdout)  # same as the preceding line
    b = (s1 == s2) and len(s1)>0
    return b
#===============================================================================================

def f1(): # a function that prints stream into the console
    print("compare this stream")
def f2(): # another function that prints stream into the console
    print("compare this stream") # you want to compare the printed text being printed out

#==============================================================================================

def solve_tower_of_hanoi(n:'number of discs',
                         source:'the starting peg',
                         destination:'destination peg',
                         intermediate:'auxilliary peg') -> 'prints the steps':
    if n==1: print("move disc %d from %s to %s" % (n, source, destination)); return;

    solve_tower_of_hanoi(n-1, source=source, destination=intermediate, intermediate=destination)
    print("move disc %d from %s to %s" % (n, source, destination))
    solve_tower_of_hanoi(n-1, source=intermediate, destination=destination, intermediate=source)

#==================================================================================================

def move_upper_section_to_intermediate(n, origin, destination, intermediate):
    solveTowerOfHanoi(n, origin, destination, intermediate)

def move_upper_section_to_destination(n, origin, destination, intermediate):
    solveTowerOfHanoi(n, origin, destination, intermediate)

def move_bottom_section_to_destination(n, origin, destination):
    print("move disc {} from {} to {}".format(n, origin, destination))

def solveTowerOfHanoi(n, origin, destination, intermediate):
    if n == 1:
        move_bottom_section_to_destination(n, origin, destination)
        return
    move_upper_section_to_intermediate(n - 1, origin=origin, destination=intermediate, intermediate=destination)
    move_bottom_section_to_destination(n, origin, destination)
    move_upper_section_to_destination(n - 1, origin=intermediate, destination=destination, intermediate=origin)

#=====================================================================================================================

def main():
    b = compare_console_output_of_two_functions(f1, f2)
    print(b)

    b = compare_console_output_of_two_functions(solve_tower_of_hanoi, solveTowerOfHanoi, 3, 'A', 'C', 'B')
    print(b)

if __name__=='__main__':main()

