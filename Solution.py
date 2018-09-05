import re
import collections

class Solution(object):
    def fullJustify(self, words, maxWidth):
        """
        :type words: List[str]
        :type maxWidth: int
        :rtype: List[str]
        """
        result = []
        i = 0
        while i < len(words):
            start = i
            words_length_in_line = 0
            words_in_line = 1
            while i<len(words) and (words_length_in_line + len(words[i]) + words_in_line - 1) <= maxWidth:
                words_length_in_line += len(words[i])
                words_in_line += 1
                i += 1
            words_in_line -= 1
            if i == len(words):
                str =''
                for k in range(start, i-1):
                    str += words[k] + ' '
                str += words[i-1]
                rest_blank = maxWidth - len(str)
                while rest_blank > 0:
                    str += ' '
                    rest_blank -= 1
                result.append(str)
                return result
            elif words_in_line == 1:
                str = words[start]
                rest_blank = maxWidth - len(str)
                while rest_blank > 0:
                    str += ' '
                    rest_blank -= 1
            else:
                allblank = maxWidth - words_length_in_line
                blank_length = int(allblank / (words_in_line - 1))
                blank_mod = allblank % (words_in_line - 1)
                str = ''
                for j in range(words_in_line):
                    str += words[start + j]
                    if j != words_in_line-1:
                        temp1 = blank_length
                        while temp1 > 0:
                            str += ' '
                            temp1 -= 1
                        if blank_mod != 0:
                            str += ' '
                            blank_mod -= 1
            result.append(str)
        return result
    
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

    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp=[]
        for i in range(n):
            if i == 0:
                dp.append(1)
            elif i == 1:
                dp.append(2)
            else:
                dp.append(dp[i-1]+dp[i-2])
        return dp[n-1]

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

    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        result_paths = []
        strs = path.split('/')
        for i in range(len(strs)):
            if strs[i] == '':
                continue
            elif strs[i] == '..':
                if len(result_paths) > 0:
                    result_paths.pop()
                else:
                    continue
            elif strs[i] == '.':
                continue
            else:
                result_paths.append(strs[i])
        result = '/'
        for i in range(len(result_paths)):
            if i != len(result_paths)-1:
                result = result + result_paths[i] + '/'
            else:
                result += result_paths[i]
        return result

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

    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        n = len(word1)+1
        m = len(word2)+1
        dpArrays = [([0] * m) for i in range(n)]
        for i in range(n):
            for j in range(m):
                if i == 0 and j == 0:
                    dpArrays[0][0] = 0
                elif i == 0:
                    dpArrays[0][j] = j
                elif j == 0:
                    dpArrays[i][0] = i
                else:
                    if word1[i-1] == word2[j-1]:
                        dpArrays[i][j] = dpArrays[i-1][j-1]
                    else:
                        dpArrays[i][j] = min(dpArrays[i][j-1], dpArrays[i-1][j], dpArrays[i-1][j-1]) + 1
        return dpArrays[n-1][m-1]

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

    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        rows = set()
        cols = set()
        m = len(matrix)
        n = len(matrix[0])
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    rows.add(i)
                    cols.add(j)
        for i in rows:
            for j in range (n):
                matrix[i][j] = 0
        for i in range (m):
            for j in cols:
                matrix[i][j] = 0
        for i in range(m):
            for j in range(n):
                print(matrix[i][j])

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        row = len(matrix)
        if row == 0:
            return False
        col = len(matrix[0])
        if col == 0:
            return False
        if target < matrix[0][0]:
            return False
        for i in range(row):
            if target > matrix[i][col-1]:
                continue
            else:
                for j in range(col):
                    if target == matrix[i][j]:
                        return True
                    else:
                        continue
        return False

    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        if not s or not t:
            return ""
        dict1 = collections.Counter(t)
        required = len(dict1)
        l = 0
        r = 0
        formed = 0
        current_window = {}
        ans = float('inf'), None, None
        while r < len(s):
            char = s[r]
            current_window[char] = current_window.get(char, 0) + 1
            if char in dict1 and current_window[char] == dict1[char]:
                formed += 1
            while l <= r and formed == required:
                if (r - l + 1) < ans[0]:
                    ans = (r - l + 1, l, r)
                char = s[l]
                l += 1
                current_window[char] = current_window[char] - 1
                if char in dict1 and current_window[char] < dict1[char]:
                    formed -= 1
            r += 1
        if ans[1] is None:
            return ""
        return s[ans[1]: ans[2]+1]

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

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """


if __name__ == '__main__':
    s = Solution()
    print(s.minWindow('abaacb', 'abc'))
