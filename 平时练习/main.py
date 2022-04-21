##汉明解法
#x = format(14,'b')
#y = int(x)
#print(y)
#print (type(y))

##子集
# def addsets(i,set):
# list.append(set)
# for j in range(i,length):
# addsets(j+1,set+[nums[j]])
#addsets(0,[])
#return list

##反转树
#root.left, root.right = root.right,root.left
# self.invertTree(root.left)
# self.invertTree(root.right)
#画图看看即可解决，区间判断再内部调整的复杂度太高，而且边界难以确定
#队列做法补充一下，广度优先
#思路是先把所有潜在的交换左右节点的root取出来压入一个队列，因为队列本身
#class Solution(object):
#	def invertTree(self, root):
#		if not root:
#			return None
		# 将二叉树中的节点逐层放入队列中，再迭代处理队列中的元素
#		queue = [root]
#		while queue:
			# 每次都从队列中拿一个节点，并交换这个节点的左右子树
#			tmp = queue.pop(0)
#			tmp.left,tmp.right = tmp.right,tmp.left
			# 如果当前节点的左子树不为空，则放入队列等待后续处理
#			if tmp.left:
#				queue.append(tmp.left)
			# 如果当前节点的右子树不为空，则放入队列等待后续处理
#			if tmp.right:
#				queue.append(tmp.right)
		# 返回处理完的根节点
#		return root

##合并二叉树
#class Solution(object):
#    def mergeTrees(self, root1, root2):
#        if root1 and root2:
#            root1.val = root1.val + root2.val
#            if root1.left and root2.left:
#                self.mergeTrees(root1.left,root2.left)
#            else:
#                if root2.left:
#                    root1.left = root2.left
#            if root1.right and root2.right:
#                self.mergeTrees(root1.right,root2.right)
#            else:
#                if root2.right:
#                    root1.right = root2.right
#        else:
#            if root2:
#                root1 = root2
#                return root1
#            else:
#                return root1
#        return root1

##比特位计算
#令 x=x & (x−1)x=x~\&~(x-1)x=x & (x−1)，该运算将 xxx 的二进制表示的最后一个 111 变成 000。因此，对 xxx 重复该操作，直到 xxx 变成 000，则操作次数即为 xxx 的「一比特数」

##排序身高问题
#按照一个维度先排列数据顺序
#sort函数：people.sort(key = lamda x:(-x[0],x[1]))
#insert函数：

##翻转图像的两种做法
#数学做法（很朴素）：
#for i in range(m):
#    for j in range(m):
#        A[j][m-i-1] = matrix[i][j]
#matrix[:] = A
#注意复制的物理id

#翻转做法：


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
"""
class Solution(object):
    def isSymmetric(self, root):
        not root:
			return True
        A = copy.deepcopy(root)
        query = [root]
        while query:
            tmp = query.pop(0)
            tmp.left,tmp.right=tmp.right,tmp.left
            if tmp.left:
                query.append(tmp.left)
            if tmp.right:
                query.append(tmp.right)
        #先反转
        #在判断是否为同一个树
        def isSametree(root1,root2):
            if root1 == root2:
                return True
            try:
                if root1.val == root2.val:
                    left = isSametree(root1.left,root2.left)
                    right = isSametree(root1.right,root2.right)
                    return left and right
            except:
                return False
            return False
        return isSametree(A,root)
"""

#树的三种遍历方式
#先序遍历：节点 左树 右树
#中序遍历：左树 节点 右树
#后序遍历：左树 右树 节点
#class Solution(object):
#    def inorderTraversal(self, root):
#       res = []
#        def mid(root):
#            if not root:
#                return []
#            else:
#                mid(root.left)
#                res.append(root.val)#位置代表了不同序的遍历,shang hou,xia qian
#                mid(root.right)
#        mid(root)

#中序遍历的迭代写法：
#        if not root:
#            return []
#        res, stack = [], []
#        while root or stack: # stack为空且root为null时，说明已经遍历结束
#            while root: # 可以深入左子树
#                stack.append(root)
#                root = root.left
#            root = stack.pop() # 否则访问栈中节点，并深入右子树
#            res.append(root.val)
#            root = root.right
#        return res
#后序遍历的迭代写法：

#迭代


#只出现一次数字
#排序后两个两个处理
#异或运算
#a^0=a,a^a=0,a^b^a=a^a^b

#前序和中序遍历得到树（前第一个是root，中序是中间root，中序和后序一样解决，后序最后一个是root，反过来做）
#记得递归的方案
#边界条件确定好需要画图，记得基准线是pref和inf，画图解决
#    def buildTree(self, preorder, inorder):
#        n = len(preorder)
#        m = len(inorder)
#        if m != n:
#            return None
#        def minitree(pref, prer, inf, inr):
#            if pref > prer or inf > inr:
#                return None
#
#            node = preorder[pref]
#            index = inf
#            while inorder[index] != node:
#                index += 1
#            root = TreeNode(node)
#            root.left = minitree(pref + 1, pref + index - inf, inf, pref + index - 1)
#            root.right = minitree(pref + index - inf + 1, prer, index + 1, inr)
#            return root
#
#        root = minitree(0, n - 1, 0, m - 1)
#        return root

#不同的搜索二叉树，第i个节点左边有i-1个数组成，右边n-i个树
#class Solution(object):
#    def numTrees(self, n):
#        a = [0 for i in range(n+1)]
#        a[0] = 1
#        for i in range(1,n+1):
#            for j in range(0,i):
#                a[i] += a[j] * a[i-j-1]
#        return a[n]

#最小路径（动态规划）
#class Solution(object):
#    def minPathSum(self, grid):
#        n = len(grid)
#        m = len(grid[0])
#        for i in range(n):
#            for j in range(m):
#                if i == j == 0:
#                    grid[i][j] = grid[i][j]
#                elif i == 0:
#                    grid[i][j] = grid[i][j-1]+ grid[i][j]
#                elif j == 0:
#                    grid[i][j] = grid[i-1][j]+ grid[i][j]
#                else:
#                    grid[i][j] = min(grid[i-1][j],grid[i][j-1])+grid[i][j]
#        return grid[-1][-1]

#每日温度##############################################################################
#第一种遍历以后的数组，没啥难度
#class Solution(object):
#    def dailyTemperatures(self, temperatures):
#        res = [0 for _ in range(len(temperatures))]
#        stack=[0]
#        for i in range(len(temperatures)):
#            if temperatures[i] <= temperatures[stack[-1]]:
#                stack.append(i)
#            else:
#                while len(stack)!=0 and temperatures[i]>temperatures[stack[-1]]:
#                    res[stack[-1]] = i-stack[-1]
#                    stack.pop()
#                stack.append(i)
#        return res

#二叉树公共祖先
#考虑中序遍历，因为中序遍历可以划分左右，且能得到根节点（否）
#考虑得到根节点到指定节点的路径，在找最近的节点（理论上可行，关键在于找到路径）
#找根-点路径似乎后序遍历可以完成，因为先遍历左右子树，不包含目标节点的话可以直接排除（worth trying）
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

#class Solution(object):
#    def lowestCommonAncestor(self, root, p, q):
#        self.res,path=[],[]
#        def dfs(node,val):
#            if not node:
#                return
#            path.append(node)
#            if node.val==val:
#                self.res=path[:]
#                return
#            if node.left:
#                dfs(node.left,val)
#                path.pop()
#            if node.right:
#                dfs(node.right,val)
#                path.pop()
#        dfs(root,q.val)###注意self.res的使用
#        a = self.res
#        print(a)
#        self.res=[]
#        path=[]
#        dfs(root,p.val)
#        b = self.res
#        print(b)
#        i = 0
#        while i<len(a) and i<len(b):
#            print(i)
#            if a[i]==b[i]:
#                i+=1
#            else:
#                break
#        return a[i-1]
#因为是递归，使用函数后可认为左右子树已经算出结果
#class Solution(object):
#    def lowestCommonAncestor(self, root, p, q):
#        if not root:
#            return root
#        if root==p or root==q:
#            return root
#        left = self.lowestCommonAncestor(root.left,p,q)
#        right = self.lowestCommonAncestor(root.right,p,q)
#        if not left:
#            return right
#        if not right:
#            return left
#        if left and right:
#            return root

#字母异位分组
#字母排序
#for i in range(len(strs)):
#            l=list(strs[i])
#            l.sort()
#            key="".join(l).encode('raw_unicode_escape')#去掉u""
#            tmp[i]=key
#class Solution(object):
#    def groupAnagrams(self, strs):
#        result = []
#        if len(strs)<=1:
#            result.append(strs)
#            return result
#        def minus(strs):
#            if len(strs)<1:
#                return
#            tmp=[]
#            index = list(strs[0])
#            index.sort()
#            index = "".join(index)
#            strs_tmp=[]
#            while len(strs):
#                key = strs.pop(0)
#                pa = list(key)
#                pa.sort()
#                pa = "".join(pa)
#                if index == pa:
#                    tmp.append(key)
#                else:
#                    strs_tmp.append(key)
#            result.append(tmp)
#            minus(strs_tmp)
#        minus(strs)
#        return result
#优化
#d=collections.defaultdict(list)#dict通过key访问变量，但是key不存在时会报错。defaultdict增加了一个可写的实例变量default_factory,
#实例变量default_factory被missing()方法使用，如果该变量存在，则用以初始化构造器，如果没有，则为None。
#class Solution(object):
#    def groupAnagrams(self, strs):
#        mp = collections.defaultdict(list)
#        for st in strs:
#            key = "".join(sorted(st))
#            mp[key].append(st)#key是sort后的string，key对应的便是结果
#        return list(mp.values())###mp.values
#对default(int)则创建一个类似dictionary对象，
#里面任何的values都是int的实例，而且就算是一个不存在的key, d[key] 也有一个默认值，这个默认值是int()的默认值0.
#计数，计算每个字母出现的次数
#mapx = collections.defaultdict(list)
#        for str in strs:
#            tmp=[0 for _ in range(26)]
#            for ch in str:
#                tmp[ord(ch)-ord("a")] +=1###ord()函数能够返回字符串的ASCII 数值，或者 Unicode 数值
#            key=tuple(tmp)
#            mapx[key].append(str)
#        return list(mapx.values())

#路径总数
#class Solution(object):
#    def uniquePaths(self, m, n):
#        mp = [[0]*n] + [[0] * (n) for _ in range(m-1)]#构建0矩阵取代zeros函数
#        for i in range(m):
#            for j in range(n):
#                if i == 0:
#                    mp[i][j]=1
#                elif j == 0:
#                    mp[i][j]=1
#                else:
#                    mp[i][j]= mp[i-1][j]+mp[i][j-1]
#        return mp[i][j]
#最简单就是组合数C （m-1，m+n-2） ，这才是随机

#链表空表构建头prehead = ListNode(-1)得到一个空链表
#        prehead = ListNode(-1)    ####重点语句 有一个重要前提是题目中规定都是大于0的正数
#        pred = prehead
#        while list1 and list2:
#            if list1.val<=list2.val:
#                pred.next = list1
#                list1=list1.next
#            elif list1.val>list2.val:
#                pred.next = list2
#                list2=list2.next
#            pred = pred.next
#        if not list1:
#            pred.next = list2
#        if not list2:
#            pred.next = list1
#        return prehead.next
#递归
#        if not list1:
#           return list2
#        if not list2:
#            return list1
#        if list1 and list2:
#            if list1.val<=list2.val:
#                list1.next = self.mergeTwoLists(list1.next,list2)
#                return list1
#            else:
#                list2.next = self.mergeTwoLists(list1,list2.next)
#                return list2

#链表排序：
#第一种读取链表的值，然后按照大小顺序赋值，主要head要保存好
#def sortList(self, head):
#        a =[]
#        head2 = head
#        while head:
#            a.append(head.val)
#            head = head.next
#        a.sort()
#        i = 0
#        head3 = head2
#        while head2:
#            head2.val = a[i]
#            head2 = head2.next
#            i+=1
#        return head3
#第二种插入排序，o(N2)
#def insertionSortList(self, head: ListNode) -> ListNode:
#    if not head:
#        return head
#    dummyHead = ListNode(0)
#    dummyHead.next = head
#    lastSorted = head
#    curr = head.next
#    while curr:
#        if lastSorted.val <= curr.val:
#            lastSorted = lastSorted.next
#        else:
#            prev = dummyHead
#            while prev.next.val <= curr.val:
#                prev = prev.next
#            lastSorted.next = curr.next
#            curr.next = prev.next
#            prev.next = curr
#        curr = lastSorted.next
#    return dummyHead.next
#第三种并归排序
#def sortList(self, head: ListNode) -> ListNode:
#    def sortFunc(head: ListNode, tail: ListNode) -> ListNode:
#        if not head:
#            return head
#        if head.next == tail:
#            head.next = None
#            return head
#        slow = fast = head#拿到三次起点
#        while fast != tail:#fast比slow要快一步
#            slow = slow.next
#            fast = fast.next
#            if fast != tail:
#                fast = fast.next
#        mid = slow
#        return merge(sortFunc(head, mid), sortFunc(mid, tail))##多看看

#    def merge(head1: ListNode, head2: ListNode) -> ListNode:
#        dummyHead = ListNode(0)
#        temp, temp1, temp2 = dummyHead, head1, head2
#        while temp1 and temp2:
#            if temp1.val <= temp2.val:
#                temp.next = temp1
#                temp1 = temp1.next
#            else:
#                temp.next = temp2
#                temp2 = temp2.next
#            temp = temp.next
#        if temp1:
#            temp.next = temp1
#        elif temp2:
#            temp.next = temp2
#        return dummyHead.next
#    return sortFunc(head, None)

#返回众数：
#    def majorityElement(self, nums):
#        if len(nums) < 1:
#            return None
#        mp = collections.defaultdict(int)
#        for i in range(len(nums)):
#            key = nums[i]
#            mp[key] += 1
#        for key in mp.keys():
#            if mp[key] > len(nums) / 2:
#                return key
#hash表
#    def majorityElemen(self, nums: List[int]) -> int:
#        counts = collections.Counter(nums)        ##该函数返回一个dict里面是key:counts
#        return max(counts.keys(), key=counts.get) ##max函数有一个key可以作为所以
#直接返回排序后中间的值一定是众数
#    class Solution(object):
#        def majorityElement(self, nums):
#            nums.sort()
#            return nums[len(nums) / 2]
#某种牛逼的做法，同加一，非同减一，找大于0的
#class Solution:
#    def majorityElement(self, nums: List[int]) -> int:
#        count = 0
#        candidate = None
#        for num in nums:
#            if count == 0:
#                candidate = num
#            count += (1 if num == candidate else -1)
#        return candidate

#回文子串
#    def countSubstrings(self, s):
#        def judge(str1,str2):
#            if str1==str2:
#                return True
#            else:
#                return False
#        def reverstr(str1):
#            return str1[::-1]
#        count=0
#        n = len(s)
#        for i in range(n):
#            length = i+1 #字符串长度
#            for j in range(n):
#                if j+length<=n:
#                    str1 = s[j:j+length]
#                    str2 = reverstr(str1)
#                    if judge(str1,str2):
#                        count+=1
#        return count
#动态规划ij的情况由【i+！】【j-1】决定当距离大于1时

#找到消失数字
#set(nums)可返回一个有序的无重复元素的set对象
#list(set(range(1,len(nums)+1))-set(nums))

#重复数字：
#mp = collections.Counter(nums)
#return max(mp.keys(),key=mp.get)

#快速排序（左右挖坑）
# 时间复杂度：o(nlogn)
#def quick_sort(alist, start, end):
#    if start >= end:
#        return
#    mid_value = alist[start]
#    low = start
#    hight = end
#    while low < hight:
#        while low<high and alist[high] >= mid_value:
#            high -= 1
#        alist[low] = alist[high]
#        while low < high and alist[low] <= mid_value:
#            low += 1
#        alist[high] = alist[low]
#    alist[low] = mid_value
#    quick_sort(alist, start, low-1)
#    quick_sort(alist, low+1, end)

#层次遍历
#    def levelOrder(self, root):
#        if not root:
#            return []
#        res = []
#
#        def dfs(root, depth):
#            if len(res) == depth:
#                res.append([])
#            res[depth].append(root.val)
#            if root.left:
#                dfs(root.left, depth + 1)
#            if root.right:
#                dfs(root.right, depth + 1)
#
#        dfs(root, 0)

#两数之和
#垃圾解法：
#str1 = 0
#str2 = 0
#i = 1
#while l1 or l2:
#    if l1:
#        str1 += l1.val * i
#        l1 = l1.next
#    if l2:
#        str2 += l2.val * i
#        l2 = l2.next
#    i *= 10
#snum = str(str1 + str2)
#head = ListNode(int(snum[-1]))
#pre = head
#for i in range(len(snum) - 1):
#    head.next = ListNode(int(snum[-i - 2]))
#    head = head.next
#return pre

#探索二维矩阵2
# 暴力查找
# 找对角线，判断不在哪个矩阵内再查找(nope,万一不是方阵)
# 每一行二分法查找
#        for row in matrix:
#            idx = bisect.bisect_left(row, target)
#            if idx < len(row) and row[idx] == target:
#                return True
#        return False

# 对角线的改进版，z形
#m, n = len(matrix), len(matrix[0])
#x, y = 0, n - 1
#while x < m and y >= 0:
#    if matrix[x][y] == target:
#        return True
#    if matrix[x][y] > target:
#        y -= 1
#    else:
#        x += 1
#return False

##########################################################################

##动态规划专题
#爬楼梯最小cost
#class Solution(object):
#    def minCostClimbingStairs(self, cost):
#        res =[]
#        res.append(cost[0])
#        res.append(cost[1])
#        n = len(cost)
#        if n>2:
#            for i in range(2,n):
#                key = min(res[i-1]+cost[i],res[i-2]+cost[i])#到第i步无论增么样都会增加一个i步的cost
#                res.append(key)
#        return min(res[-1],res[-2])

#整数拆分
#class Solution(object):
#    def integerBreak(self, n):
#        res = [0] * (n+1)
#        res[0]=1
#        res[1]=1
#        res[2]=1
#        for i in range(3,n+1):
#            for j in range(1,i-1):
#                res[i] = max(res[i],max(j*(i-j),j*res[i-j]))##j*(i-j)因为再多分出一个1结果一定比这个小
#        return res[n]
#
#单词拆分###############################################切分单词和第二个循环的判断值得体会
#dp = [False]*(len(s) + 1)
#        dp[0] = True
        # 遍历背包
#        for j in range(1, len(s) + 1):
            # 遍历单词
#            for word in wordDict:#相当于背包问题里的重量，后面的长度为价值
#                if j >= len(word):##j小于单词长度没必要看了
#                    dp[j] = dp[j] or (dp[j - len(word)] and word == s[j - len(word):j])
#                    ##j可是是false，j为true只有在上一个节点也是true的情况下，word等于词典内的word
#        return dp[len(s)]
#乘积最大子数组（连续子数组意思是物理位置连续而不是，物理位置和逻辑位置都联系，逻辑位置只需要加一个判断即可）
#resmax维护结果，防止负数的存在使得结果最大可能反转
#        resmax = [0 for i in range(len(nums))]
#        resmin = [0 for i in range(len(nums))]
#        resmax[0],resmin[0] = nums[0],nums[0]
#        for i in range(1,len(nums)):
#            resmax[i] = max(resmax[i-1]*nums[i],max(nums[i],resmin[i-1]*nums[i]))
#            resmin[i] = min(resmin[i-1]*nums[i],min(nums[i],resmax[i-1]*nums[i]))
#        return max(resmax)
##########
#最大正方形
##########
#方法一讨论每一个值为1的格子，向下和向右的m-i，n-j的节点包含的正方形的值是否为m+n-i-j
#方法二，节点是否为1，是1的话，则由左，上，左上节点中最小值决定res值，最终的结果是res【i】最大者的平方respect！！！
#        m = len(matrix)
#        n = len(matrix[0])
#        self.maxarea = 0
#        for i in range(m):
#            for j in range(n):
#                if i == 0 or j==0:
#                    if matrix[i][j]=="1" and self.maxarea==0:
#                        self.maxarea =1
#                elif i !=0 and j!=0 and matrix[i][j]=="1":
#                    matrix[i][j] = str(min(int(matrix[i-1][j]),int(matrix[i][j-1]),int(matrix[i-1][j-1]))+1)
#                    area = int(matrix[i][j]) * int(matrix[i][j])
#                    if area >= self.maxarea:
#                        self.maxarea = area
#        return self.maxarea

#零钱兑换（返回最小硬币数量）
#        res = [float("inf") for i in range(amount+1)]
#        res[0] = 0
#        for i in range(1,amount+1):
#            for j in range(len(coins)):
#                if i-coins[j]>=0:
#                    res[i] = min(res[i],res[i-coins[j]]+1)
#        if res[-1]==float("inf"):
#            return -1
#        return res[-1]

#零钱兑换（返回组合数量）

#打家劫舍三
#def trackval(root):
#    if not root:
#        return 0, 0
#    leftchild_steal, leftchild_nosteal = trackval(root.left)
#    rightchild_steal, rightchild_nosteal = trackval(root.right)
    # 偷当前node，则最大收益为【投当前节点+不偷左右子树】
#    steal = root.val + leftchild_nosteal + rightchild_nosteal
    # 不偷当前node，则可以偷左右子树
#    nosteal = max(leftchild_steal, leftchild_nosteal) + max(rightchild_steal, rightchild_nosteal)
#    return steal, nosteal
#return max(trackval(root))










###################################################################################################
#
#盛最多水的容器
#双指针
#    l,r=0,len(height)-1
#    ans=0
#    while l < r:
#        area = min(height[l], height[r]) * (r - l)
#        ans = max(ans, area)
#        if height[l] <= height[r]:
#            l += 1
#        else:
#            r -= 1
#    return ans

#55跳跃游戏动规（fall）
#if not nums:
#    return False
#if len(nums) == 1:
#    return True
#if len(nums) >= 2:
#    res = [False for _ in range(len(nums))]
#    res[0] = True
#    for i in range(len(nums)):
#        for j in range(i + 1):
#            if res[j] == True and nums[j] >= i - j:
#                res[i] = True
#                break
#return res[-1]
#贪心
#n, rightmost = len(nums), 0
#        for i in range(n):
#            if i <= rightmost:
#                rightmost = max(rightmost, i + nums[i])
#                if rightmost >= n - 1:
#                    return True
#        return False

#荷兰国旗
#先统计个数，按序排列
#hashtable = collections.Counter(nums)
#        for i in range(hashtable[0]):
#            nums[i]=0
#        for i in range(hashtable[1]):
#            nums[hashtable[0]+i]=1
#        for i in range(hashtable[2]):
#            nums[hashtable[0]+hashtable[1]+i]=2

#判断二叉搜索树
#中序遍历的结果一定是递增的
#递归
#        if not root:
#            return True
#        self.result = True
#        def judge(root,left,right):
#            if root.val<=left or root.val>=right:
#                self.result = False
#            if root.left:
#                judge(root.left,left,root.val)
#            if root.right:
#                judge(root.right,root.val,right)
#        judge(root,float('-inf'),float('inf'))
#        return self.result

#三数之和
#if len(nums)<=2:
    # return []
    # result=[]
    #nums.sort()
#        for i in range(len(nums)):
#            target=-nums[i]#两数之和的目标值
#            hashtable = dict()
#            for j,num in enumerate(nums[i+1:]):##########################################################两数之和
#                tmp=[]
#                if target - num in hashtable.values():#如果在hash表内则输出
#                    tmp.append(nums[i])
#                    tmp.append(num)
#                    tmp.append(target - num)
#                    #tmp.sort()#增加一个很小的时间复杂去重
#                hashtable[j] = num
#                if tmp and tmp not in result:#以防万一写的去重
#                    result.append(tmp)
#        return result

#删除链表其中一个节点
#    def removeNthFromEnd(self, head, n):
#        dummy = ListNode(0, head)  # 删除头节点的细节
#        first = head
#        second = dummy
#        for i in range(n):
#            first = first.next
#        while first:
#            first = first.next
#            second = second.next
#        second.next = second.next.next
#        return dummy.next

#下一个排列
#for i in range(len(nums)-1,0,-1):
#            if nums[i-1] < nums[i]:
#                for j in range(len(nums)-1,i-1,-1):
#                    if nums[j] > nums[i-1]:
#                        nums[i-1],nums[j] = nums[j],nums[i-1]
#                        break
#                for j in range((len(nums)-i+1)//2):
#                    nums[i+j],nums[len(nums)-1-j] = nums[len(nums)-1-j] ,nums[i+j]
#                return nums
#        nums.reverse()
#        return nums
#
#买卖股票的最佳时机
#动规
#        mini, maxProfit = prices[0], 0
#        for i in range(1,len(prices)):
#            maxProfit = max(maxProfit, prices[i] - mini)
#            mini = min(mini, prices[i])
#        return maxProfit
#贪心

#最长连续序列
#       longest_streak = 0
#        num_set = set(nums)
#        for num in num_set:
#            if num - 1 not in num_set:
#                current_num = num
#                current_streak = 1
#                while current_num + 1 in num_set:
#                    current_num += 1
#                    current_streak += 1
#                longest_streak = max(longest_streak, current_streak)
#        return longest_streak

###############环形链表###################
#第一种构建一个hashtable，判断节点是否在hash表内
#        seen = set()
#        while head:
#            if head in seen:
#                return True
#            seen.add(head)
#            head = head.next
#        return False

#
#龟兔赛跑
#快指针比慢指针快两步，快指针在环内一定能追上慢指针是true，快指针到null则是false
####相交链表###############注意一个很奇怪的点set（）时间比数组快很多？？？？？？？
#        meet = set()
#        result = None
#        while headA:
#            meet.add(headA)
#            headA = headA.next
#        while headB:
#            if headB not in meet:
#                meet.add(headB)
#                headB = headB.next
#            else:
#                result =headB
#                break
#        return result
#双指针：两个头节点走相同长度的路，要么都是none要么相遇
#        if not headA or not headB:
#            return None
#        nodeA = headA
#        nodeB = headB
#        while(nodeA !=nodeB):
#            nodeA = nodeA.next if nodeA else headB
#            nodeB = nodeB.next if nodeB else headA
#        return nodeA
####环形指针：
#        if not head:
#            return None
#        if not head.next and head:
#            return None
#        self.slow, self.fast, self.index = head, head, head
#        while self.fast and self.fast.next:
#            self.slow = self.slow.next
#            self.fast = self.fast.next.next
#            if self.slow == self.fast:
#                while self.index != self.slow:
#                    self.index = self.index.next
#                    self.slow = self.slow.next
#                return self.index
#            elif not self.fast:
#                return None

#########################树的深度优先搜索最大直径
#        self.ans = 1
#        def depth(node):
#            if not node:
#                return 0
#            L = depth(node.left)
#            R = depth(node.right)
#            self.ans = max(self.ans, L + R + 1)
#            return max(L, R) + 1
#        depth(root)
#        return self.ans - 1
##字符串解码（以后遇到括号的题多往栈靠）
#        stack = []
#        for i in s:
#            if i == ']':
#                strs = ''
#                repeat = ''
#                while stack[-1] != '[':
#                    strs = stack.pop() + strs
#                stack.pop()
#                while stack and stack[-1].isdigit():###判断是否为数字
#                    repeat = stack.pop() + repeat
#                stack.append(int(repeat) * strs)
#                continue
#            stack.append(i)
#        return ''.join(stack)



##找到字符串中所有异味字符串位置
#维护一个n大小的数组去和目标数组比较，右加入时对应位数加一，同时左减一对应字符数量，比求字典快很多
#        m, n = len(s), len(p)
#        ans = []
#        if m < n:
#            return ans
#        cnts_p = [0] * 26
#        for c in p:
#            cnts_p[ord(c) - ord('a')] += 1
#        cur = [0] * 26
#        for i in range(m):
#            cur[ord(s[i]) - ord('a')] += 1
#            if i >= n - 1:
#                if cur == cnts_p:
#                    ans.append(i - n + 1)
#                cur[ord(s[i - n + 1]) - ord('a')] -= 1
#        return ans

#双指针解决数组删除特定数字或重复数字的方案
#       if not nums:
#            return 0
#        n = len(nums)
#        fast = slow = 0
#        while fast < n:
#            if nums[fast] != val:
#                nums[slow] = nums[fast]
#                slow += 1
#            fast += 1
#        return slow

#重排链表 #翻转链表和插入链表结合
#        dummy = ListNode(0,head)
#        pre, mid, count, index = head, head, head, 0
#        while count:
#            index += 1
#            count = count.next
#        if index % 2 == 0:
#            tail = 1+(index-1)/2
#        else:
#            tail = (index-1)/2
#        length = index
#        index = 0
#        while index<tail:
#            index += 1
#            mid = mid.next
#        tail = length-tail-1
#        index = 0
#        def reversenode(head):
#            pre, cur = None, head
#            while cur:
#                next = cur.next
#                cur.next = pre
#                pre = cur
#                cur = next
#            return pre
#        mid.next = reversenode(mid.next)
#        while index<tail:
#            pnext = pre.next
#            midnext = mid.next
#            mid.next = midnext.next
#            pre.next = midnext
#            midnext.next = pnext
#            pre = midnext.next
#            index +=1