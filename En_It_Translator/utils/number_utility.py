def is_ordinal_number(word):
    try:
        int(word.replace('th', ''))
        return True
    except ValueError:
        return False


def is_number(word):
    try:
        float(word)
        return True
    except ValueError:
        return False


def int_to_roman(input):
    """ Convert an integer to a Roman numeral. """

    if not isinstance(input, type(1)):
        return False
    if not 0 < input < 4000:
        return False
    ints = (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)
    nums = ('M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I')
    result = []
    for i in range(len(ints)):
        count = int(input / ints[i])
        result.append(nums[i] * count)
        input -= ints[i] * count
    return ''.join(result)


def is_roman_number(input):
    """ Convert a Roman numeral to an integer. """
    if not isinstance(input, type("")):
        return False
    input = input.upper()
    nums = {'M': 1000, 'D': 500, 'C': 100, 'L': 50, 'X': 10, 'V': 5, 'I': 1}
    sum = 0
    for i in range(len(input)):
        try:
            value = nums[input[i]]
            # If the next place holds a larger number, this value is negative
            if i + 1 < len(input) and nums[input[i + 1]] > value:
                sum -= value
            else:
                sum += value
        except KeyError:
            return False
    # easiest test for validity...
    if int_to_roman(sum) == input:
        return True
    else:
        return False
