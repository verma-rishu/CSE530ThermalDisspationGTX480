import re
import sys

GPU_AVG_REGEX = r'gpu_avg_[A-Z0-9_]*,? = \d+\.*\d*'
GPU_MIN_REGEX = r'gpu_min_[A-Z0-9_]*,? = \d+\.*\d*'
GPU_MAX_REGEX = r'gpu_max_[A-Z0-9_]*,? = \d+\.*\d*'
SEARCH_RESULT_SPLIT_SEPARATOR = ', = '
SEARCH_RESULT_SPLIT_SEPARATOR_ALT = ' = '
MAP_UNIT_KEY = 'UNIT'
MAP_VALUE_KEY = 'VALUE'
#PTRACE_FILENAME = 'output512.ptrace'

header = [  'L0DRAM1', 'L0DRAM2', 'L0DRAM3','L0DRAM4','L0DRAM5','L0DRAM6', 'L0GT','L0HI','PLACEHLDR',
            'DRAM1', 'DRAM2', 'DRAM3','DRAM4','DRAM5','DRAM6',
            'GT','HI','L2C',
            'UC1','UC2','UC3','UC4','UC5','UC6','UC7','UC8','UC9','UC10','UC11','UC12','UC13','UC14','UC15','UC16',
            'SHDMEM1','SHDMEM2','SHDMEM3','SHDMEM4','SHDMEM5','SHDMEM6','SHDMEM7','SHDMEM8','SHDMEM9','SHDMEM10','SHDMEM11','SHDMEM12','SHDMEM13','SHDMEM14','SHDMEM15','SHDMEM16',
            'ICN1','ICN2','ICN3','ICN4','ICN5','ICN6','ICN7','ICN8','ICN9','ICN10','ICN11','ICN12','ICN13','ICN14','ICN15','ICN16',
            'LDST1','LDST2','LDST3','LDST4','LDST5','LDST6','LDST7','LDST8','LDST9','LDST10','LDST11','LDST12','LDST13','LDST14','LDST15','LDST16',
            'SFU1','SFU2','SFU3','SFU4','SFU5','SFU6','SFU7','SFU8','SFU9','SFU10','SFU11','SFU12','SFU13','SFU14','SFU15','SFU16',
            'RF1','RF2','RF3','RF4','RF5','RF6','RF7','RF8','RF9','RF10','RF11','RF12','RF13','RF14','RF15','RF16',
            'IC1','IC2','IC3','IC4','IC5','IC6','IC7','IC8','IC9','IC10','IC11','IC12','IC13','IC14','IC15','IC16',
            'SM1C1','SM1C2','SM2C1','SM2C2','SM3C1','SM3C2','SM4C1','SM4C2','SM5C1','SM5C2','SM6C1','SM6C2','SM7C1','SM7C2','SM8C1','SM8C2','SM9C1','SM9C2','SM10C1','SM10C2','SM11C1','SM11C2','SM12C1','SM12C2','SM13C1','SM13C2','SM14C1','SM14C2','SM15C1','SM15C2','SM16C1','SM16C2',
            'SM1WS1','SM1WS2','SM2WS1','SM2WS2','SM3WS1','SM3WS2','SM4WS1','SM4WS2','SM5WS1','SM5WS2','SM6WS1','SM6WS2','SM7WS1','SM7WS2','SM8WS1','SM8WS2','SM9WS1','SM9WS2','SM10WS1','SM10WS2','SM11WS1','SM11WS2','SM12WS1','SM12WS2','SM13WS1','SM13WS2','SM14WS1','SM14WS2','SM15WS1','SM15WS2','SM16WS1','SM16WS2',
            'SM1DU1','SM1DU2','SM2DU1','SM2DU2','SM3DU1','SM3DU2','SM4DU1','SM4DU2','SM5DU1','SM5DU2','SM6DU1','SM6DU2','SM7DU1','SM7DU2','SM8DU1','SM8DU2','SM9DU1','SM9DU2','SM10DU1','SM10DU2','SM11DU1','SM11DU2','SM12DU1','SM12DU2','SM13DU1','SM13DU2','SM14DU1','SM14DU2','SM15DU1','SM15DU2','SM16DU1','SM16DU2']

def process_plog_file(filepath: str, regex: str) -> dict:
    with open(filepath, 'r') as file:
        file_content = file.read()
        matched_values = re.findall(regex, file_content)
        # extracted_values_list = []
        extracted_values_list = {}
        for value in matched_values:
            split_value = value.split(SEARCH_RESULT_SPLIT_SEPARATOR)
            try:
                checkSplit = split_value[1]
            except IndexError:
                split_value = value.split(SEARCH_RESULT_SPLIT_SEPARATOR_ALT)
            unit, value = split_value[0][8:], float(split_value[1])
            extracted_values_list[unit] = value
            # extracted_values_list.append({MAP_UNIT_KEY: unit, MAP_VALUE_KEY: value})
        return extracted_values_list

def component_power(output):
    comp_pwr = {}
    # for pair in output:
    for MAP_UNIT_KEY, MAP_VALUE_KEY in output.items():
        if MAP_UNIT_KEY == 'DRAMP' or MAP_UNIT_KEY == 'MCP':
            if 'DRAM' not in comp_pwr: 
                comp_pwr['DRAM'] = MAP_VALUE_KEY
            else:
                comp_pwr['DRAM'] += MAP_VALUE_KEY
        elif MAP_UNIT_KEY == 'L2CP':
            if 'L2C' not in comp_pwr: 
                comp_pwr['L2C'] = MAP_VALUE_KEY
            else:
                comp_pwr['L2C'] += MAP_VALUE_KEY
        elif MAP_UNIT_KEY == 'CCP':
            if 'UC' not in comp_pwr: 
                comp_pwr['UC'] = MAP_VALUE_KEY
            else:
                comp_pwr['UC'] += MAP_VALUE_KEY
        elif MAP_UNIT_KEY == 'SHRDP' or MAP_UNIT_KEY == 'DCP':
            if 'SHDMEM' not in comp_pwr: 
                comp_pwr['SHDMEM'] = MAP_VALUE_KEY
            else:
                comp_pwr['SHDMEM'] += MAP_VALUE_KEY
        elif MAP_UNIT_KEY == 'NOCP':
            if 'ICN' not in comp_pwr: 
                comp_pwr['ICN'] = MAP_VALUE_KEY
            else:
                comp_pwr['ICN'] += MAP_VALUE_KEY
        elif MAP_UNIT_KEY == 'SPP' or MAP_UNIT_KEY == 'FPUP' or MAP_UNIT_KEY == 'PIPEP' or MAP_UNIT_KEY == 'COREP' or MAP_UNIT_KEY == 'IDLE_COREP' or MAP_UNIT_KEY == 'CONST_DYNAMICP': 
            if 'CORE' not in comp_pwr: 
                comp_pwr['CORE'] = MAP_VALUE_KEY
            else:
                comp_pwr['CORE'] += MAP_VALUE_KEY
        elif MAP_UNIT_KEY == 'SFUP':
            if 'SFU' not in comp_pwr: 
                comp_pwr['SFU'] = MAP_VALUE_KEY
            else:
                comp_pwr['SFU'] += MAP_VALUE_KEY
        elif MAP_UNIT_KEY == 'RFP':
            if 'RF' not in comp_pwr: 
                comp_pwr['RF'] = MAP_VALUE_KEY
            else:
                comp_pwr['RF'] += MAP_VALUE_KEY
        elif MAP_UNIT_KEY == 'SCHEDP':
            if 'WS' not in comp_pwr: 
                comp_pwr['WS'] = MAP_VALUE_KEY
            else:
                comp_pwr['WS'] += MAP_VALUE_KEY
        elif MAP_UNIT_KEY == 'ICP' or MAP_UNIT_KEY == 'IBP':
            if 'IC' not in comp_pwr: 
                comp_pwr['IC'] = MAP_VALUE_KEY
            else:
                comp_pwr['IC'] += MAP_VALUE_KEY
    return comp_pwr

def createPtrace(PTRACE_FILENAME):
    with open(PTRACE_FILENAME, 'w') as ptrace: 
            ptrace.writelines(['\t'.join(header[:-1]) + '\t' + header[-1] + '\n'])

def addValuesPtrace(comp_pwr, PTRACE_FILENAME):
    ptrace_line = {}
    ptrace_line['GT']=0
    ptrace_line['HI']=0
    ptrace_line['L0GT']=0
    ptrace_line['L0HI']=0
    ptrace_line['PLACEHLDR']=0
    for i in range(1, 17):
        ptrace_line['LDST' + str(i)] = 0

    for comp in comp_pwr.keys():
            if comp == 'DRAM':
                for i in range(1, 7):
                    ptrace_line['DRAM' + str(i)] = (comp_pwr[comp])/6
                    ptrace_line['L0DRAM' + str(i)] = (comp_pwr[comp])/6
                    #ptrace_values['MEM' + str(i)] = ptrace_values['DRAM' + str(i)]
            elif comp == 'L2C':
                ptrace_line['L2C'] = comp_pwr[comp]
            elif comp == 'UC':
                for i in range(1, 17):
                    ptrace_line['UC' + str(i)] = comp_pwr[comp]/16
            elif comp == 'SHDMEM':
                for i in range(1, 17):
                    ptrace_line['SHDMEM' + str(i)] = comp_pwr[comp]/16
            elif comp == 'ICN':
                for i in range(1, 17):
                    ptrace_line['ICN' + str(i)] = comp_pwr[comp]/16
            elif comp == 'IC':
                for i in range(1, 17):
                    ptrace_line['IC' + str(i)] = comp_pwr[comp]/16
            elif comp == 'SFU':
                for i in range(1, 17):
                    ptrace_line['SFU' + str(i)] = comp_pwr[comp]/16
            elif comp == 'RF':
                for i in range(1, 17):
                    ptrace_line['RF' + str(i)] = comp_pwr[comp]/16
            elif comp == 'CORE':
                for i in range(1, 17):
                    ptrace_line['SM' + str(i)+'C1'] = comp_pwr[comp]/32
                    ptrace_line['SM' + str(i)+'C2'] = comp_pwr[comp]/32
            elif comp == 'WS':
                for i in range(1, 17):
                    unit_pwr = comp_pwr[comp]/64
                    ptrace_line['SM' + str(i)+'WS1'] = unit_pwr
                    ptrace_line['SM' + str(i)+'WS2'] = unit_pwr
                    ptrace_line['SM' + str(i)+'DU1'] = unit_pwr
                    ptrace_line['SM' + str(i)+'DU2'] = unit_pwr
    ptrace_line = {key: ptrace_line[key] for key in header}
    with open(PTRACE_FILENAME, 'a') as ptrace:
        line = ''
        for key in header[:-1]:
            line += (str(ptrace_line[key]) + '\t')
        line += (str(ptrace_line[header[-1]]) + '\n')
        ptrace.writelines(line)

def main():
    if len(sys.argv) != 3:
        print('Input name of the .log file as the argument as: python logToPtrace.py <logFile> <outputFileName>')
        sys.exit(0)
    PTRACE_FILENAME = sys.argv[2]
    output_avg = process_plog_file(sys.argv[1],GPU_AVG_REGEX)
    output_min = process_plog_file(sys.argv[1],GPU_MIN_REGEX)
    output_max = process_plog_file(sys.argv[1],GPU_MAX_REGEX)
    comp_pwr_avg = component_power(output_avg)
    comp_pwr_min = component_power(output_min)
    comp_pwr_max = component_power(output_max)
    createPtrace(PTRACE_FILENAME)
    addValuesPtrace(comp_pwr_avg, PTRACE_FILENAME)
    addValuesPtrace(comp_pwr_min, PTRACE_FILENAME)
    addValuesPtrace(comp_pwr_max, PTRACE_FILENAME)

if __name__ == '__main__':
    main()