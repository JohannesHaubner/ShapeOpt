import re

from pathlib import Path
here = Path(__file__).parent.parent.resolve()

def generate_table(txtname : str, txtout : str, txtout_all : str):

    text_file = open(str(here) + '/mesh/Output/' + txtname,'r')

    # lines into separate strings
    file_lines = text_file.readlines()

    # list of line starts
    start = ['Objective', 'Dual', 'Constraint', 'Variable', 'Complementarity', 'Overall', 'Number', 'Total', 'EXIT', 'Warning']

    start_ov_wo_reg = ['Objective value']

    start_num = []
    for i in range(10):
        for j in ['', ' ', '  ', '   ','    ']:
            start_num.append(j+str(i))

    important_lines = []
    data = []
    header_ipopt = []
    ob_val = []
    all = []
    for line in file_lines:

        flag = False
        flag_num = False
        flag_header = False
        flag_obval = False

        for i in start:
            if re.search('\A'+i, line):
                flag = True
        for i in start_num:
            if re.search('\A'+i, line):
                flag_num = True
        if re.search('\A'+'iter', line):
                flag_header = True
        
        for i in start_ov_wo_reg:
            if re.search('\A'+i, line):
                flag_obval = True

        if flag:
            important_lines.append(line)
            all.append(line)
        if flag_num:
            data.append(line)
            all.append(line)
        if flag_header:
            header_ipopt.append(line)
            all.append(line)
        if flag_obval:
            ob_val.append(line)
            all.append(line)

    def reduce_list(ell):
        ell = list(dict.fromkeys(ell))
        ell = [i[:-1] for i in ell]
        return ell

    def split_at_whitespace(ell):
        ell = [[i for j in k.split() for i in (j, ' ')] for k in ell] 
        return ell

    def clean_ipopt_data(ell):
        data_ipopt = []
        for i in ell:
            if len(i)==20:
                if i[12]=='-':
                    data_ipopt.append(i)
        return data_ipopt

    def remove_whitespace_from_list(ell):
        ell = [list(filter((' ').__ne__, k)) for k in ell]
        return ell
    
    def add_previous_line_objective(ell):
        new_list = []
        old_line = []
        for i in ell:
            if i[2] == str('at'):
                new_list.append(old_line)
            else:
                old_line = i
        return new_list

    data_list = [important_lines, data, header_ipopt, ob_val, all]
    data_list = [reduce_list(i) for i in data_list] # remove duplicates and \n at end
    for i in [1,2,3]:
        data_list[i] = split_at_whitespace(data_list[i])
    data_list[1] = clean_ipopt_data(data_list[1]) # since lines coming from IPOPT output typically have 20 sub-strings at this point and entry data[i][12] == '-' - remove all other lines that do not fulfill this
    for i in [1,2,3]:
        data_list[i] = remove_whitespace_from_list(data_list[i]) # remove all 

    # postprocess data_list[3] remove all lines that are close to being identity 
    data_list[3] = add_previous_line_objective(data_list[3])


    data_array = []
    #data = data_list[2]
    #data.append('objective_w/o_reg_pen')
    #data_array.append(data)
    for i in range(len(data_list[1])):
        data = data_list[1][i]
        if i <= len(data_list[3])-1:
            data.append(data_list[3][i][-1])
        else:
            data.append(' ')
        data_array.append(data)

    import os.path
    file_txt = os.path.join(str(here) + '/visualization/', txtout)

    file_all_txt = os.path.join(str(here) + '/visualization/', txtout_all)

    with open(file_all_txt, 'w') as file:
        for i in range(len(data_list[4])):
            file.write(data_list[4][i] + '\n')

    # write into txt.file which contains latex table code
    with open(file_txt, 'w') as file:
        file.write("\\begin{table}[ht!]\n")
        file.write("  \\begin{tabularx}{\\textwidth}{R|x|x|x|xe}\n")
        file.write("    \\arrayrulecolor{white}\n")
        file.write("    \\rowcolor{tumblues2}\n")
        file.write("    \\textcolor{white}{iteration} & \\textcolor{white}{objective}& \\textcolor{white}{objective \mbox{ \small w/o reg. \& pen.}}& \\textcolor{white}{dual infeasibility} & \\textcolor{white}{linesearch-steps} &\\\\[0.5ex]\n")
        for i in range(len(data_array)):
            if i > 1:
                file.write("%s & $%s$ & $%s$ & $%s$ & $%s$ & \\\\[0.5ex]\n" % (data_array[i][0], data_array[i][1], data_array[i][-1], data_array[i][3], data_array[i][-2]))
            else:
                file.write("%s & $%s$ & $%s$ & $%s$ & $%s$ & \\\\[0.5ex]\n" % (data_array[i][0], data_array[i][1], data_array[i][-1], data_array[i][3], data_array[i][-2]))
            if i%2 == 0:
                file.write("    \\rowcolor{tumg}\n")
        file.write("  \\end{tabularx}\n")
        file.write("  \\caption{Optimization results when IPOPT converges up to an overall NLP tolerance of $10^{-3}$}\n")
        file.write("  \\label{tab::or2}\n")
        file.write("\\end{table}")

    pass


if __name__ == "__main__":
    txtname = "terminal_2207.txt"
    txtout = 'table.txt'
    txtout_all = 'to_shortened.txt'
    generate_table(txtname, txtout, txtout_all)