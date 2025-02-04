# Replace '0' by '1' and '1' by '0' (can be solved easily with only one state)
xs1 = ["010101", "1100101", "001"]
ys1 = ["101010", "0011010", "110"]
alphabet_in_1 = ['0', '1']
alphabet_out_1 = ['0', '1']

# Add 1 to each digit (can be solved with only one state)
xs2 = ["001234", "5214110", "334"]
ys2 = ["112345", "0325221", "445"]
alphabet_in_2 = ['0', '1', '2', '3', '4', '5']
alphabet_out_2 = ['0', '1', '2', '3', '4', '5']

# Emit always the first digit seen
xs3 = ["03021211231", "101131122", "2013100", "32331100", "0123", "2", "211"]
ys3 = ["00000000000", "111111111", "2222222", "33333333", "0000", "2", "211"]
alphabet_in_3 = ['0', '1', '2', '3']
alphabet_out_3 = ['0', '1', '2', '3']

# Replace each character by the amount of consecutive repetitions
xs4 = ["ababab", "aaababba", "bbbbaaaa", "aabbaaa"]
ys4 = ["111111", "12311121", "12341234", "1212123"]
alphabet_in_4 = ['a', 'b']
alphabet_out_4 = ['1', '2', '3', '4']

# Emit 0 and 1 alternately without care the input
xs5 = ["abccbcadjig", "abcdefghij", "aabcigf", "aah", "a"]
ys5 = ["10101010101", "1010101010", "1010101", "101", "1"]
alphabet_in_5 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
alphabet_out_5 = ['0', '1']

# Replace the apparitions of '101' for 'aaa'
xs6 = ["10100", "00110010", "101010101", "111", "1", "10"]
ys6 = ["aaa00", "00110010", "aaa0aaa01", "111", "1", "10"]
alphabet_in_6 = ['0', '1']
alphabet_out_6 = ['0', '1', 'a']

# Roman to decimal
xs7 = ["III#", "IV#", "VI#", "IX#", "XI#", "XIX#", "XLII#", "LXXIV#", "XCV#", "CXV#", "CDII#", "CMLXXXIX#", "MCXI#", "MMMMMMMMDCCCLXXXVIII#"]
ys7 = ["3###", "4##", "6##", "9##", "11#", "19##", "42###", "74####", "95##", "115#", "402##", "989######", "1111#", "8888#################"]
alphabet_in_7 = ["I", "V", "X", "L", "C", "D", "M", "#"]
alphabet_out_7 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '#']

# Emit 0 but emit 1 when consume the first 1
xs8 = ["000100", "00110010", "101010000", "111", "0", "00001001"]
ys8 = ["000111", "00111111", "111111111", "111", "0", "00001111"]
alphabet_in_8 = ['0', '1']
alphabet_out_8 = ['0', '1']

# Replace 'a' by 'zxy'
xs9 = ["abbbb", "aa", "bbbbbabb", "bbbbbbb", "bb", "abbbabbbb", "bbba"]
ys9 = ["zxybb", "zx", "bbbbbzxy", "bbbbbbb", "bb", "zxybzxybb", "bbbz"]
alphabet_in_9 = ['a', 'b']
alphabet_out_9 = ['a', 'b', 'z', 'x', 'y']

# Right shift by 1 bit
xs10 = ["010010101", "101", "000010101", "11011011", "1", "011001001111", "0111"]
ys10 = ["001001010", "010", "000001010", "01101101", "0", "001100100111", "0011"]
alphabet_in_10 = ['0', '1']
alphabet_out_10 = ['0', '1']