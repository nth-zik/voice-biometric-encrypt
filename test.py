import graycode
import math
def float_to_gray_with_sign(value, decimal_keep):
    """Chuyển đổi float thành Gray code với 1 bit dấu và 63 bit giá trị."""
    sign_bit = "0" if value >= 0 else "1"  # 0 cho dương, 1 cho âm
    exponent = int(math.log10(decimal_keep))

    value_rounded = round(value, exponent)
    abs_value = abs(value) * decimal_keep  # Nhân với hằng số để loại bỏ phần thập phân
    int_value = int(abs_value)  # Chuyển thành số nguyên
    binary_str = str(int_value)
    binary_str_total = ''
    for s in binary_str:
        gray_value = '{:05b}'.format(graycode.tc_to_gray_code(int(s)))
        print(s, gray_value)
        binary_str_total = binary_str_total + str(gray_value)
    # binary_str_total = '{:063b}'.format(binary_str_total)
    # Đảm bảo chuỗi nhị phân có độ dài 63 bit
    total_value_bits = 63
    if len(binary_str_total) > total_value_bits:
        # Cắt bớt chuỗi nếu dài hơn 63 bit
        binary_str_total = binary_str_total[:total_value_bits]
    elif len(binary_str_total) < total_value_bits:
        # Đệm thêm các số 0 ở cuối nếu ngắn hơn 63 bit
        binary_str_total = binary_str_total.rjust(total_value_bits, '0')
    print("{:063b}".format(graycode.tc_to_gray_code(int_value)))
    print(binary_str_total)
    return sign_bit + binary_str_total  # 1 bit dấu + 63 bit giá trị Gray code


float_to_gray_with_sign(1.1036871075454, 10**8)
