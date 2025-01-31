# Replace '0' by '1' and '1' by '0'
xs1 = ["010101", "1100101", "001"]
ys1 = ["101010", "0011010", "110"]
alphabet_in_1 = ['0', '1']
alphabet_out_1 = ['0', '1']


# Add 1 to each digit
xs2 = ["001234", "5214110", "334"]
ys2 = ["112345", "0325221", "445"]
alphabet_in_2 = ['0', '1', '2', '3', '4', '5']
alphabet_out_2 = ['0', '1', '2', '3', '4', '5']

# For each digit  repeated twice replace by the next and prev digit
xs3 = ["0155221244", "001133442", "12344100"]
ys3 = ["0140131235", "510224352", "12335151"]
alphabet_in_3 = ['0', '1', '2', '3', '4', '5']
alphabet_out_3 = ['0', '1', '2', '3', '4', '5']

# Replace 'a' for 'b'
xs4 = ["ababab", "bbabab", "bba"]
ys4 = ["bababa", "aababa", "aab"]

# Roan to decimal
xs5 = ["III#", "IV#", "VI#", "IX#", "XI#", "XIX#", "XLII#", "LXXIV#", "XCV#", "CXV#", "CDII#", "CMLXXXIX#", "MCXI#", "MMMMMMMMDCCCLXXXVIII#"]
ys5 = ["3###", "4##", "6##", "9##", "11#", "19##", "42###", "74####", "95##", "115#", "402##", "989######", "1111#", "8888#################"]