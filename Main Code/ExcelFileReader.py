# python script that will read an excel file and print the contents
# to the console
import xlrd

workbook = xlrd.open_workbook('dataSet1.xlsx')

sheet = workbook.sheet_by_index(0)

num_rows = sheet.nrows

num_cols = sheet.ncols

for curr_row in range(num_rows):
    for curr_col in range(num_cols):
        print(sheet.cell_value(curr_row, curr_col))
