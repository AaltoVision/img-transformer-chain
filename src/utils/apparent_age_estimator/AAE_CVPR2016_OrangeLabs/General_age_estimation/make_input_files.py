from os import listdir
from os.path import isfile, join
import sys

if len(sys.argv) < 2:
	print("Not enough arguments")
	sys.exit()

INPUT_FILES_DIR = sys.argv[1]

def get_file_list(mypath):
    return [(f, join(mypath, f)) for f in listdir(mypath) if isfile(join(mypath, f))]

file_list = get_file_list(INPUT_FILES_DIR)

max_files = 2000
cycle = 50
current_file_i = 1
current_line = 1

print('Maximum number of files per directory is {}'.format(max_files))

for name_line, dir_line in file_list:
    try:
        if current_line == 1:
            fn1 = "../img_pointers_{}".format(current_file_i)
            fn2 = "../img_names_{}".format(current_file_i)
            print("Open next files... {} {}".format(fn1, fn2))
            ptr_file = open(fn1, "w")
            name_file = open(fn2, "w")

        ptr_file.write(dir_line + '\n')
        name_file.write(name_line + '\n')

        current_line = current_line + 1
        if current_line > cycle:
            print("Writing done.")
            current_file_i = current_file_i+1

            if current_file_i > max_files:
                break
            
            ptr_file.close()
            name_file.close()

            print("Files closed.")

            current_line = 1
    except:
        print("Some error. Skipped that line.")

ptr_file.close()
name_file.close()
print("Files closed.")
