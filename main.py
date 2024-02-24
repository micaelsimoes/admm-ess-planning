import os
import sys
import getopt
from energy_storage_planning import EnergyStoragePlanning

# ======================================================================================================================
#  Read Execution Arguments
# ======================================================================================================================
def read_execution_arguments(argv):

    test_case_dir = str()
    spec_filename = str()

    try:
        opts, args = getopt.getopt(argv, 'hd:f:', ['help', 'test_case=', 'specification_file='])
    except getopt.GetoptError:
        print_help_message()
        sys.exit(2)

    if not argv or not opts:
        print_help_message()
        sys.exit()

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print_help_message()
            sys.exit()
        elif opt in ('-d', '--dir'):
            test_case_dir = arg
        elif opt in ('-f', '--file'):
            spec_filename = arg

    if not test_case_dir or not spec_filename:
        print('Shared Resource Planning Tool usage:')
        print('\tmain.py -d <test_case> -f <specification_file>')
        sys.exit()

    return spec_filename, test_case_dir


def print_help_message():
    print('\nShared Resources Planning Tool usage:')
    print('\n Usage: main.py [OPTIONS]')
    print('\n Options:')
    print('   {:25}  {}'.format('-d, --test_case=', 'Directory of the Test Case to be run (located inside "data" directory)'))
    print('   {:25}  {}'.format('-f, --specification_file=', 'Specification file of the Test Case to be run (located inside the "test_case" directory)'))
    print('   {:25}  {}'.format('-h, --help', 'Help. Displays this message'))


# ======================================================================================================================
#  Energy Storage Planning
# ======================================================================================================================
def energy_storage_planning(working_directory, specification_filename):

    print('==========================================================================================================')
    print('                                    ENERGY STORAGE PLANNING (BENDERS\')                                   ')
    print('==========================================================================================================')

    planning_problem = EnergyStoragePlanning(working_directory, specification_filename)
    planning_problem.read_planning_problem()

    planning_problem.run_operational_planning()
    #planning_problem.run_planning_problem()



# ======================================================================================================================
#  Main
# ======================================================================================================================
if __name__ == '__main__':
    filename, test_case = read_execution_arguments(sys.argv[1:])
    directory = os.path.join(os.getcwd(), 'data', test_case)
    energy_storage_planning(directory, filename)
