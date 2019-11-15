class NumberUtility:
    def is_ordinal_number(self, word):
        try:
            int(word.replace('th', ''))
            return True
        except ValueError:
            return False

    def is_number(self, word):
        try:
            float(word)
            return True
        except ValueError:
            return False

    def __int_to_roman(self, num):
        """ Convert an integer to a Roman numeral. """
        if not isinstance(num, type(1)):
            return False
        if not 0 < num < 4000:
            return False
        ints = (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)
        nums = ('M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I')
        result = []
        for i in range(len(ints)):
            count = int(num / ints[i])
            result.append(nums[i] * count)
            num -= ints[i] * count
        return ''.join(result)

    def is_roman_number(self, num):
        """ Convert a Roman numeral to an integer. """
        if not isinstance(num, type("")):
            return False
        num = num.upper()
        nums = {'M': 1000, 'D': 500, 'C': 100, 'L': 50, 'X': 10, 'V': 5, 'I': 1}
        sum = 0
        for i in range(len(num)):
            try:
                value = nums[num[i]]
                # If the next place holds a larger number, this value is negative
                if i + 1 < len(num) and nums[num[i + 1]] > value:
                    sum -= value
                else:
                    sum += value
            except KeyError:
                return False
        # easiest test for validity...
        return self.__int_to_roman(sum) == num
