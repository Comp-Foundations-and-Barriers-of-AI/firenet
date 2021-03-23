from os.path import join

def read_count(fname='COUNT.txt', count_path='./'):
    """ Read and updates the runner count. 
    
    To keep track of all the different runs of the algorithm, one store the 
    run number in the file 'COUNT.txt' at ``count_path``. It is assumed that 
    the file 'COUNT.txt' is a text file containing one line with a single 
    integer, representing number of runs so far. 

    This function reads the current number in this file and increases the 
    number by 1. 
    
    :return: Current run number (int).
    """
    fname = join(count_path, fname);
    infile = open(fname);
    data = infile.read();
    count = int(eval(data));
    infile.close();

    outfile = open(fname, 'w');
    outfile.write('%d ' % (count+1));
    outfile.close();
    return count;

