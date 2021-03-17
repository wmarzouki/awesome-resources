import csv
import sys
import re

OUTPUT_FILENAME = "README.md"


def clean(string):
    """
    Remove new lines from a string
    """
    return re.sub(r'\n{1,}', " ", string)




def generate_content(filename):
    """
    Generate a markdown table and write it to
    README.md file
    """
    table = "| Topic  | Small description | Link | \n |-------|-------------------|-----|\n"

    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        reader.__next__() # escape first row
        for row in reader:
            table += f"| {clean(row[0])} | {clean(row[1])} | {clean(row[2])} | \n"

    with open(OUTPUT_FILENAME, "w") as outfile:
        outfile.write(table)

    

if __name__ == "__main__" :
    if sys.argv[1]:
        generate_content(sys.argv[1])
    else:
        raise OSError("filename is missing")

