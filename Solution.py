import re
import collections
import ListNode
import Stack
import copy
import TreeNode


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
        result = []
        i = 0
        result.append([])
        while i <len(nums):
            for j in range(len(result)):
                currentlist = []
                for k in range(len(result[j])):
                    currentlist.append(result[j][k])
                currentlist.append(nums[i])
                result.append(currentlist)
            i += 1
        return result

    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        if len(board) ==0 or len(board[0]) == 0 or len(word) == 0 or (len(board) * len(board[0]) < len(word)):
            return False
        visited = []
        for i in range(len(board)):
            tmp = []
            for j in range(len(board[0])):
                tmp.append(False)
            visited.append(tmp)
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.exist_rec(board, word, 0, i, j,visited):
                    return True
        return False

    def exist_rec(self, board, word, index, x, y, visited):
        if index == len(word):
            return True
        if x < 0 or x == len(board) or y < 0 or y == len(board[0]) or visited[x][y]:
            return False
        if board[x][y] == word[index]:
            visited[x][y] = True
            up = self.exist_rec(board, word, index+1, x-1, y, visited)
            down = self.exist_rec(board, word, index+1, x+1, y, visited)
            left = self.exist_rec(board, word, index+1, x, y-1, visited)
            right = self.exist_rec(board, word, index+1, x, y+1, visited)
            visited[x][y] = False
            return down | up | left | right
        else:
            return False

    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) <= 2:
            return len(nums)
        length = 0
        currentlen = 0
        i = 0
        while i < len(nums):
            if i == 0:
                length += 1
                currentlen += 1
            else:
                if nums[i] == nums[length-1]:
                    currentlen += 1
                    if currentlen <= 2:
                        if i != length and nums[length] != nums[i]:
                            nums[length] = nums[i]
                        length += 1
                    else:
                        j = i
                        while j < len(nums) and nums[j] == nums[length-1]:
                            j += 1
                        if j < len(nums):
                            nums[length] = nums[j]
                            i = j
                            length += 1
                            currentlen = 1
                else:
                    currentlen = 1
                    if i != length and nums[length] != nums[i]:
                        nums[length] = nums[i]
                    length += 1
            i += 1
        return length

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        def search_inside(left, right):
            if left > right:
                return False
            mid = int((left + right) / 2)
            if nums[mid] == target:
                return True
            elif nums[mid] > target:
                return search_inside(left, mid-1)
            else:
                return search_inside(mid+1, right)

        head = 0
        for i in range(1, len(nums)):
            if nums[i] < nums[i - 1]:
                head = i
                break
        if head == 0:
            return search_inside(0, len(nums) - 1)
        else:
            if target > nums[0]:
                return search_inside(1, head-1)
            elif target < nums[0]:
                return search_inside(head, (len(nums)-1))
            else:
                return True


    ### ！虽然是medium 但是满难的！
    def deleteDuplicates2(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return head
        newnode = ListNode.ListNode(0)
        newnode.next = head
        pre = newnode
        while pre.next:
            cur = pre.next
            while cur.next and cur.val == cur.next.val:
                cur = cur.next
            if pre.next == cur:
                pre = cur
            else:
                pre.next = cur.next
        return newnode.next

    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        cur = head
        while cur:
            if cur.next and cur.val == cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return head

    def largestRectangleArea1(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        res = 0
        h = copy.copy(heights)
        h.append(0)
        for i in range(len(h)):
            if i+1 == len(h) or h[i] <= h[i+1]:
                continue
            else:
                minHeight = h[i]
                for j in range(i, -1, -1):
                    minHeight = min(h[j], minHeight)
                    area = minHeight * (i-j+1)
                    res = max(res, area)
        return res

    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        stack = Stack.Stack()
        i = 0
        maxarea = 0
        h = copy.copy(heights)
        h.append(0)
        while i < len(h):
            if stack.is_empty() or h[stack.peek()] <= h[i]:
                stack.push(i)
                i += 1
            else:
                t = stack.pop()
                width = 0
                if stack.is_empty():
                    width = i
                else:
                    width = i - stack.peek() -1
                maxarea = max(maxarea, h[t] * width)
        return maxarea

    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        heights = []
        for i in range(len(matrix)):
            current_row_height = []
            for j in range(len(matrix[0])):
                h = 0
                row = i
                while row >= 0 and matrix[row][j] == '1':
                    h += 1
                    row -= 1
                current_row_height.append(h)
            heights.append(current_row_height)
        maxarea = 0
        for i in range(len(matrix)):
            maxarea = max(maxarea, self.largestRectangleArea(heights[i]))
        return maxarea

    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        answer = None
        if not head:
            return answer
        answerlast = answer
        largerlist = None
        largerlist_pre = largerlist
        current = head
        while current:
            if current.val < x:
                if answer is None:
                    answer = ListNode.ListNode(current.val)
                    answerlast = answer
                else:
                    node = ListNode.ListNode(current.val)
                    answerlast.next = node
                    answerlast = node
            else:
                if largerlist is None:
                    largerlist = ListNode.ListNode(current.val)
                    largerlist_pre = largerlist
                else:
                    node = ListNode.ListNode(current.val)
                    largerlist_pre.next = node
                    largerlist_pre = node
            current = current.next
        if answer:
            answerlast.next = largerlist
        else:
            return head
        return answer

    def isScramble(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        if len(s1) != len(s2):
            return False
        if s1 == s2:
            return True
        ss1 = "".join((lambda x: (x.sort(), x)[1])(list(s1)))
        ss2 = "".join((lambda x: (x.sort(), x)[1])(list(s2)))
        if ss1 != ss2:
            return False
        for i in range(1, len(s1)):
            s11 = s1[0:i]
            s12 = s1[i:len(s1)]
            s21 = s2[0:i]
            s22 = s2[i:len(s2)]
            if self.isScramble(s11, s21) and self.isScramble(s12, s22):
                return True
            elif self.isScramble(s11, s22) and self.isScramble(s12, s21):
                return True
            s21 = s2[0:len(s2)-i]
            s22 = s2[len(s2)-i:len(s2)]
            if self.isScramble(s11, s21) and self.isScramble(s12, s22):
                return True
            elif self.isScramble(s11, s22) and self.isScramble(s12, s21):
                return True
        return False

    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        if not nums1 or n == 0:
            return
        i = 0
        ri = 0
        j = 0
        while ri < m and j < n:
            while ri < m and nums1[i] <= nums2[j]:
                i += 1
                ri += 1
            if ri < m:
                nums1.insert(i, nums2[j])
                nums1.pop()
                i += 1
                j += 1
        while i < len(nums1) and j < n:
            nums1[i] = nums2[j]
            i += 1
            j += 1

    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        res = []
        for i in range(2**n):
            res.append(i >> 1 ^ i)
        return res

    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        res.append([])
        for i in range(len(nums)):
            for j in range(len(res)):
                current = []
                current_ans = res[j]
                for k in range(len(current_ans)):
                    current.append(current_ans[k])
                current.append(nums[i])
                current.sort()
                if current not in res:
                    res.append(current)
        return res

    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        dp = []
        if len(s) == 0 or s[0] == '0':
            return 0
        elif len(s) == 1:
            return 1
        dp.append(1)
        for i in range(1, len(s)):
            if i == 1:
                if s[i] == '0':
                    if s[0] == '1' or s[0] == '2':
                        dp.append(1)
                    else:
                        dp.append(0)
                else:
                    if s[0] == '1' or (s[0] == '2' and s[1] <= '6'):
                        dp.append(2)
                    else:
                        dp.append(1)
            else:
                if s[i-1] == '0' and s[i] == '0':
                    dp.append(0)
                else:
                    if s[i] == '0':
                        if s[i-1] == '1' or s[i-1] == '2':
                            dp.append(dp[i-2])
                        else:
                            dp.append(0)
                    else:
                        if s[i-1] == '1' or (s[i-1] == '2' and s[i] <= '6'):
                            dp.append(dp[i-1]+dp[i-2])
                        else:
                            dp.append(dp[i-1])
        return dp[len(s)-1]

    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        if m == n:
            return head
        res_node = ListNode.ListNode(0)
        res_node_last = res_node
        index = 1
        current = head
        restnode = None
        while current:
            if index == m:
                while index < n:
                    current = current.next
                    index += 1
                restnode = current.next
                temp = head
                index = 1
                while index < m:
                    temp = temp.next
                    index += 1
                prenode = ListNode.ListNode(temp.val)
                temp = temp.next
                index += 1
                if n-m > 1:
                    nextnode = temp.next
                    temp = ListNode.ListNode(temp.val)
                    while index < n:
                        temp.next = prenode
                        prenode = temp
                        temp = ListNode.ListNode(nextnode.val)
                        nextnode = nextnode.next
                        index += 1
                temp.next = prenode
                res_node_last.next = temp
                while temp.next:
                    temp = temp.next
                temp.next = restnode
                return res_node.next
            else:
                res_node_last.next = ListNode.ListNode(current.val)
                res_node_last = res_node_last.next
                current = current.next
                index += 1

    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        if len(s) < 4 or len(s) > 12:
            return []
        result = []
        self.restoreIpAddress_rec(s, 0, 0, '', result)
        return result

    def restoreIpAddress_rec(self, s, index, ipindex, current, result):
        if ipindex == 4 and index == len(s):
            result.append(current)
            return
        if len(s)-index < (4 - ipindex):
            return
        if len(s)-index >= 1:
            str1 = s[index:index + 1]
            if current == '':
                str = str1
            else:
                str = current + '.' + str1
            self.restoreIpAddress_rec(s, index+1, ipindex+1, str, result)
        if len(s)-index >= 2:
            str2 = s[index:index + 2]
            if str2[0] == '0':
                return
            if current == '':
                str = str2
            else:
                str = current + '.' + str2
            self.restoreIpAddress_rec(s, index+2, ipindex+1, str, result)
        if len(s)-index >= 3:
            str3 = s[index:index + 3]
            if str3[0] == '0':
                return
            if int(str3) < 256:
                if current == '':
                    str = str3
                else:
                    str = current + '.' + str3
                self.restoreIpAddress_rec(s, index+3, ipindex+1, str, result)

    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        result = []
        current = root
        while current:
            if not current.left:
                result.append(current.val)
                current = current.right
            else:
                pre = current.left
                while pre.right and pre.right != current:
                    pre = pre.right
                if not pre.right:
                    pre.right = current
                    current = current.left
                else:
                    pre.right = None
                    result.append(current.val)
                    current = current.right
        return result

    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        dpArray = [0, 1]
        index = 2
        while index <= n:
            current = 0
            for i in range(1, (index+1)):
                if i == 1 or i == index:
                    current += dpArray[index-1]
                else:
                    current += dpArray[index-i]*dpArray[i-1]
            dpArray.append(current)
            index += 1
        return dpArray[n]

    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        if n == 0:
            return []
        return self.generateTrees_rec(1, n)

    def generateTrees_rec(self, left, right):
        allroots = []
        if left>right:
            allroots.append(None)
            return allroots
        if left == right:
            allroots.append(TreeNode.TreeNode(left))
            return allroots
        for i in range(left, right+1):
            leftchildnodes = self.generateTrees_rec(left, i-1)
            rightchildnodes = self.generateTrees_rec(i+1, right)
            for j in range(len(leftchildnodes)):
                for k in range(len(rightchildnodes)):
                    root = TreeNode.TreeNode(i)
                    root.left = leftchildnodes[j]
                    root.right = rightchildnodes[k]
                    allroots.append(root)
        return allroots

    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        m = len(s1)
        n = len(s2)
        if (m+n) != len(s3):
            return False
        dpArrays = [([False] * (n+1)) for i in range(m+1)]
        dpArrays[0][0]=True
        for i in range(m):
            if s1[i] == s3[i] and dpArrays[i][0]:
                dpArrays[i+1][0] = True
        for j in range(n):
            if s2[j] == s3[j] and dpArrays[0][j]:
                dpArrays[0][j+1] = True
        for i in range(1, m+1):
            for j in range(1, n+1):
                if (s1[i-1]==s3[i+j-1] and dpArrays[i-1][j]) or (s2[j-1] == s3[i+j-1] and dpArrays[i][j-1]):
                    dpArrays[i][j] = True
        return dpArrays[m][n]

    pre = None
    first = None
    second = None

    def recoverTree(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        self.recoverTree_inorder(root)
        if self.first is not None and self.second is not None:
            temp = self.first.val
            self.first.val = self.second.val
            self.second.val = temp
            print(root.val)

    def recoverTree_inorder(self, root):
        if root is None:
            return
        self.recoverTree_inorder(root.left)
        if self.pre is None:
            self.pre = root
        else:
            if self.pre.val > root.val:
                if self.first is None:
                    self.first = self.pre
                self.second = root
            self.pre = root
        self.recoverTree_inorder(root.right)

    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        return self.isValidBST_rec(root, float("-inf"), float("inf"))

    def isValidBST_rec(self, node, lower, upper):
        if node.val <= lower or node.val >= upper:
            return False
        if node.left and not self.isValidBST_rec(node.left, lower, node.val):
                return False
        if node.right and not self.isValidBST_rec(node.right, node.val, upper):
                return False
        return True

    def isSameTree(self, p, q):
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) & self.isSameTree(p.right, q.right)

    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if not postorder or not inorder:
            return None
        root = TreeNode.TreeNode(postorder[len(postorder)-1])
        root_index = 0
        for i in inorder:
            if i != root.val:
                root_index += 1
            else:
                break
        left_postorder = postorder[:root_index]
        right_postorder = postorder[root_index:len(postorder)-1]
        left_inorder = inorder[:root_index]
        right_inorder = inorder[root_index+1:len(inorder)]
        if left_inorder and left_postorder:
            root.left = self.buildTree(left_inorder, left_postorder)
        if right_inorder and right_postorder:
            root.right = self.buildTree(right_inorder, right_postorder)
        return root

    def buildTree1(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if not preorder or not inorder:
            return None
        root = TreeNode(preorder[0])
        root_index = 0
        for i in inorder:
            if i != preorder[0]:
                root_index += 1
            else:
                break
        left_preorder = preorder[1:(1+root_index)]
        right_preorder = preorder[(1+root_index):len(preorder)]
        left_inorder = inorder[:root_index]
        right_inorder = inorder[(1+root_index):len(inorder)]
        if left_preorder and left_inorder:
            root.left = self.buildTree(left_preorder, left_inorder)
        if right_preorder and right_inorder:
            root.right = self.buildTree(right_preorder, right_inorder)
        return root

    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        result = []
        if not root:
            return result
        queue = []
        queue.append(root)
        while queue:
            templist = []
            tempresult = []
            for i in queue:
                templist.append(i)
                tempresult.append(i.val)
            queue = []
            result.append(tempresult)
            for i in templist:
                if i.left:
                    queue.append(i.left)
                if i.right:
                    queue.append(i.right)
        result.reverse()
        return result

    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return None
        mid = int(len(nums)/2)
        root = TreeNode.TreeNode(nums[mid])
        if len(nums) > 1:
            left_nums = nums[:mid]
            root.left = self.sortedArrayToBST(left_nums)
        if len(nums) > 2:
            right_nums = nums[mid+1:len(nums)]
            root.right = self.sortedArrayToBST(right_nums)
        return root


if __name__ == '__main__':
    s = Solution()
    root = s.sortedArrayToBST([-10, -3, 0, 5, 9])
    print(root.val)