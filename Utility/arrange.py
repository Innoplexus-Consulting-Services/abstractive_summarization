"""script to arrange the ref* and decode* output when textsum is
    run in decode mode."""
import sys
import os

#decode,ref and arranged_output filename
decoded_file = 'decode'
reference_file = 'reference'
arranged_file = 'decode_sample'

count = 0

#reads and arranges the output in 'arrange_file'
def _read_and_arrange_write(ref,dec,arr):
    with open(ref,'r') as file_ref, open(dec,'r') as file_dec:
        dec_lines = file_dec.readlines()
        ref_lines = file_ref.readlines()
        ref_sum = [''] * (len(dec_lines) + 1)
        ref_sum[-1] = 'output='
        index = -1
        for line in ref_lines:
            if index == len(dec_lines):
                print(index)
                break
            line = line.strip()
            if type(line) == 'NoneType':
                continue
            if line.startswith('output=') and len(line) > 7:
                index += 1
                line = line[7:]
            if line.startswith('output=') and len(line) == 7:
                break
            # print(type(ref_sum[index]), type(line))
            ref_sum[index] += ' ' + line

        with open(arr,'w') as f:
            for index in range(len(dec_lines)):
                f.write('\n*********ACTUAL(' + str(index) +')******\n')
                f.write(ref_sum[index])
                f.write('\n*********GENERATED(' + str(index) +')******\n')
                f.write(dec_lines[index])
                f.write('-`-----------------------------------------------------------------------------------------------------------')

if __name__ = '__main__':
    _read_and_arrange_write(reference_file,decoded_file,arranged_file)
