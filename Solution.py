import re

class Solution(object):
    def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if len(s) == 0:
            return False
        regex = r'[+-]?(\d+\.?|\.\d+)\d*(e[+-]?\d+)?'
        s = s.strip()
        result = re.match(regex, s)
        if result:
            print("matchObj.group() : ", result.group())
        else:
            return False

    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        carry = 0
        toadd = 1
        for i in range(len(digits) - 1, -1, -1):
            if toadd == 1:
                digits[i] += 1
                toadd = 0
            if carry == 1:
                digits[i] += 1
                carry = 0

            if digits[i] == 10:
                digits[i] = 0
                carry = 1
            if carry == 0:
                break
        if carry == 1:
            digits.insert(0, 1)

        return digits

    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        answer = ''
        if len(a) == 0 & len(b) == 0:
            return answer
        if len(a) == 0:
            return b
        elif len(b) == 0:
            return a
        if len(a) > len(b):
            for i in range(len(a)-len(b)):
                b = '0' + b
        elif len(a) < len(b):
            for i in range(len(b)-len(a)):
                a = '0' + a
        carry = 0
        for i in range(len(a)-1, -1, -1):
            n1 = int(a[i])
            n2 = int(b[i])
            if (n1 + n2 + carry) == 0:
                answer = '0' + answer
                carry = 0
            elif (n1 + n2 + carry) == 1:
                answer = '1' + answer
                carry = 0
            elif (n1 + n2 + carry) == 2:
                answer = '0' + answer
                carry = 1
            elif (n1 + n2 + carry) == 3:
                answer = '1' + answer
                carry = 1
        if carry == 1:
            answer = '1' + answer
        return answer

    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x <= 1:
            return x
        left = 0
        right = x
        while (right - left) != 1:
            mid = int((left + right)/2)
            if mid * mid > x:
                right = mid
            elif mid * mid < x:
                left = mid
            else:
                return mid
        if right * right > x:
            return left
        else:
            return right

    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        length = len(nums)
        for i in range(length):
            if nums[i] != 0:
                for j in range(length-1, i-1, -1):
                    if nums[j] == 0:
                        tmp = nums[j]
                        nums[j] = nums[i]
                        nums[i] = tmp
                        break
                if j == i:
                    if nums[i] == 2:
                        for j in range(length - 1, i-1, -1):
                            if nums[j] == 1:
                                tmp = nums[j]
                                nums[j] = nums[i]
                                nums[i] = tmp
                                break
            else:
                continue

    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        if len(s) < len(t):
            return ""
        dict = {}
        start = -1
        end = -1
        min = len(s)
        hasAll = False
        for i in range(len(t)):
            dict[t[i]] = -1
        for j in range(len(s)):
            if s[j] in dict:
                if len(dict) == 1:
                    end = j
                    start = j
                    break
                else:
                    dict[s[j]] = j
                    temp = len(s)
                    for key in dict:
                        if key != s[j]:
                            if dict[key] == -1:
                                hasAll = False
                                break
                            hasAll = True
                            if dict[key] < temp:
                                temp = dict[key]
                    if hasAll and j-temp < min:
                        min = j - temp
                        start = temp
                        end = j
        if end == -1:
            return ""
        return s[start:end+1]

    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        result = []
        if n == 0 or k == 0:
            return result
        list = []
        self.combine_rec(1, list, result, n ,k)
        return result

    def combine_rec(self, index, list, result, n, k):
        if len(list) == k:
            r = []
            for i in range(len(list)):
                r.append(list[i])
            result.append(r)
            return
        for i in range(index, n+1):
            list.append(i)
            self.combine_rec(i+1, list, result, n, k)
            list.pop()


if __name__ == '__main__':
    solution = Solution()
    print(solution.combine(4, 2))
