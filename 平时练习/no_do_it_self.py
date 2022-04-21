#79单词搜索可以再做一次
#directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
#def check(i, j, k):
#    if board[i][j] != word[k]:
#        return False
#    if k == len(word) - 1:
#        return True
#    visited.add((i, j))
#    result = False
#    for di, dj in directions:
#        newi, newj = i + di, j + dj
#        if 0 <= newi < len(board) and 0 <= newj < len(board[0]):
#            if (newi, newj) not in visited:
#                if check(newi, newj, k + 1):
#                    result = True
#                    break
#    visited.remove((i, j))
#    return result
#h, w = len(board), len(board[0])
#visited = set()
#for i in range(h):
#    for j in range(w):
#        if check(i, j, 0):
#            return True
#return False

# class DLinkedNode:
#     def __init__(self, key=0, value=0):
#         self.key = key
#         self.value = value
#         self.prev = None
#         self.next = None
#
#
# class LRUCache:
#
#     def __init__(self, capacity: int):
#         self.cache = dict()
#         # 使用伪头部和伪尾部节点
#         self.head = DLinkedNode()
#         self.tail = DLinkedNode()
#         self.head.next = self.tail
#         self.tail.prev = self.head
#         self.capacity = capacity
#         self.size = 0
#
#     def get(self, key: int) -> int:
#         if key not in self.cache:
#             return -1
#         # 如果 key 存在，先通过哈希表定位，再移到头部
#         node = self.cache[key]
#         self.moveToHead(node)
#         return node.value
#
#     def put(self, key: int, value: int) -> None:
#         if key not in self.cache:
#             # 如果 key 不存在，创建一个新的节点
#             node = DLinkedNode(key, value)
#             # 添加进哈希表
#             self.cache[key] = node
#             # 添加至双向链表的头部
#             self.addToHead(node)
#             self.size += 1
#             if self.size > self.capacity:
#                 # 如果超出容量，删除双向链表的尾部节点
#                 removed = self.removeTail()
#                 # 删除哈希表中对应的项
#                 self.cache.pop(removed.key)
#                 self.size -= 1
#         else:
#             # 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
#             node = self.cache[key]
#             node.value = value
#             self.moveToHead(node)
#
#     def addToHead(self, node):
#         node.prev = self.head
#         node.next = self.head.next
#         self.head.next.prev = node
#         self.head.next = node
#
#     def removeNode(self, node):
#         node.prev.next = node.next
#         node.next.prev = node.prev
#
#     def moveToHead(self, node):
#         self.removeNode(node)
#         self.addToHead(node)
#
#     def removeTail(self):
#         node = self.tail.prev
#         self.removeNode(node)
#         return node

#课程表
        #思想：节点一共有三种状态：【未访问1】【为结束探索1】【已结束探索2】意味着需要有一个数组去维护访问状态（visited）
        #每次从未访问的节点开始，dfs其【用一个prerequisites【1，0】的hashtable访问】所有节点并改状态为1
        #如果是未访问，dfs，直至其没有节点（无后续节点或者后续节点状态为2）可以访问，则入栈，并且该状态为2
        #如果遇到的是1节点则改判断为false
#        edges = collections.defaultdict(list)
#        visited = [0] * numCourses
        #result = list()似乎没有啥用
#        self.valid = True

#        for info in prerequisites:
#            edges[info[1]].append(info[0])

#        def dfs(u):
#            visited[u] = 1
#            for v in edges[u]:
#                if visited[v] == 0:
#                    dfs(v)
#                    if not self.valid:
#                        return       #return的用法可以借鉴一下，结束当前函数，理论上和break作用类似
#                elif visited[v] == 1:
#                    self.valid = False
#                    return
#            visited[u] = 2
#            #result.append(u)

#        for i in range(numCourses):
#            if self.valid and not visited[i]:
#                dfs(i)

#        return self.valid

#买卖股票2，涉及到多次买卖和冷冻期
#if not prices:
#    return 0

#n = len(prices)
# f[i][0]: 手上持有股票的最大收益
# f[i][1]: 手上不持有股票，并且处于冷冻期中的累计最大收益
# f[i][2]: 手上不持有股票，并且不在冷冻期中的累计最大收益
#f = [[-prices[0], 0, 0]] + [[0] * 3 for _ in range(n - 1)]
#for i in range(1, n):
#    f[i][0] = max(f[i - 1][0], f[i - 1][2] - prices[i])
#    f[i][1] = f[i - 1][0] + prices[i]
#    f[i][2] = max(f[i - 1][1], f[i - 1][2])

#return max(f[n - 1][1], f[n - 1][2])

#字符串匹配思路：两种情况匹配到* or not 没懂的一点就是为什么是按位或
#p遇到.和字符直接判断，相等则【i，j】结果等于【i-1，j-1】的结果；
#遇到*有两种可能，*不匹配，【i，j】等于【i，j-2】结果，*匹配，判断（s[i],p[j-1])，如果匹配则【i，j】为【i-1，j】的值
#class Solution(object):
#    def isMatch(self, s, p):
#        m, n = len(s), len(p)
#
#        def matches(i, j):
#            if i == 0:
#                return False
#            if p[j - 1] == '.':
#                return True
#            return s[i - 1] == p[j - 1]
#
#        f = [[False] * (n + 1) for _ in range(m + 1)]
#        f[0][0] = True
#        for i in range(m + 1):
#            for j in range(1, n + 1):
#                if p[j - 1] == '*':
#                    f[i][j] |= f[i][j - 2]#两者都为false时，值为false，otherwise equal true
#                    if matches(i, j - 1):
#                        f[i][j] |= f[i - 1][j]
#                else:
#                    if matches(i, j):
#                        f[i][j] |= f[i - 1][j - 1]
#        return f[m][n]
#最长有效括号
#动态规划
#        if not s:
#            return 0
#        n = len(s)
#        dp = [0]*n
#        for i in range(len(s)):
#            if s[i]==')' and i-dp[i-1]-1>=0 and s[i-dp[i-1]-1]=='(':
#               dp[i]=dp[i-1]+dp[i-dp[i-1]-2]+2
#        return max(dp)
#栈
#        stack = [-1]
#        ret = 0
#        lg = len(s)
#        for i in range(lg):
#            if s[i] == '(':
#                stack.append(i)
#            else:
#                stack.pop()
#                if not stack:
#                    stack.append(i)
#                else:
#                    ret = max(ret, i - stack[-1])
#        return ret

#接雨水（动态规划）
#if not height:
#    return 0

#leftmax = [0 for _ in range(len(height))]
#leftmax[0] = height[0]
#for i in range(1, len(height)):
#    leftmax[i] = max(height[i], leftmax[i - 1])
#
#rightmax = [0 for _ in range(len(height))]
#rightmax[-1] = height[-1]
#for i in range(len(height) - 2, -1, -1):
#    rightmax[i] = max(height[i], rightmax[i + 1])
#return sum(min(rightmax[i], leftmax[i]) - height[i] for i in range(len(height)))


#####编辑距离
#        if not word1 and not word2:
#            return 0
#        if not word1:
#            return len(word2)
#        if not word2:
#            return len(word1)
#        n = len(word1)
#        m = len(word2)
#        res = [[0 for _ in range(n+1)] for i in range(m+1)]
#        for i in range(n+1):
#            res[0][i]=i
#        for j in range(m+1):
#            res[j][0]=j
#        for i in range(1,m+1):
#            for j in range(1,n+1):
#                if word1[j-1] == word2[i-1]:
#                    res[i][j] = min(res[i-1][j],res[i][j-1],(res[i-1][j-1]-1)) + 1
#                else:
#                    res[i][j] = min(res[i-1][j],res[i][j-1],res[i-1][j-1]) + 1
#        return res[m][n]
#覆盖最小子串
#        if not s or len(s)<len(t):
#            return ""
#        if s == t:
#            return t
#        def haveset(set1,set2):
#            for key in set1.keys():
#                if set1[key] > set2[key]:
#                    return False
#            return True
#        t = str(t)
#        target = collections.Counter(t)
#        for i in range(0,len(s)):
#            for j in range(len(s)):
#                if haveset(target,collections.Counter(s[j:j+i+1])) and j+i+1<=len(s):
#                    return s[j:j+i+1]
#        return ""
#柱状图中的最大矩形
        # 注意n为非负整数
        # 思路就是找固定高度的矩形最长的延展距离
#        n = len(heights)
#        left, right = [0] * n, [0] * n
#        mono_stack = list()
#        for i in range(n):
#            while mono_stack and heights[mono_stack[-1]] >= heights[i]:
#                mono_stack.pop()
#            left[i] = mono_stack[-1] if mono_stack else -1
#            mono_stack.append(i)
#        mono_stack = list()
#        for i in range(n - 1, -1, -1):
#            while mono_stack and heights[mono_stack[-1]] >= heights[i]:
#                mono_stack.pop()
#            right[i] = mono_stack[-1] if mono_stack else n
#            mono_stack.append(i)
#        ans = max((right[i] - left[i] - 1) * heights[i] for i in range(n)) if n > 0 else 0
#        return ans


#删除无效括号并返回不重复的结果：
#        res = []
#        lremove, rremove = 0, 0
#        for c in s:
#            if c == '(':
#                lremove += 1
#            elif c == ')':
#                if lremove == 0:
#                    rremove += 1
#                else:
#                    lremove -= 1

#        def isValid(str):
#            cnt = 0
#            for c in str:
#                if c == '(':
#                    cnt += 1
#                elif c == ')':
#                    cnt -= 1
#                    if cnt < 0:
#                        return False
#            return cnt == 0

#        def helper(s, start, lremove, rremove):
#            if lremove == 0 and rremove == 0:
#                if isValid(s):
#                    res.append(s)
#                return



#            for  i in range(start, len(s)):
#                if i > start and s[i] == s[i - 1]:
#                    continue
                #### 去重操作，相同的符号只需要去一个其他同理（即去掉最右那个连续相同括号）####
                # 如果剩余的字符无法满足去掉的数量要求，直接返回
#                if lremove + rremove > len(s) - i:
#                    break
                # 尝试去掉一个左括号
#                if lremove > 0 and s[i] == '(':
#                    helper(s[:i] + s[i + 1:], i, lremove - 1, rremove);
                # 尝试去掉一个右括号
#                if rremove > 0 and s[i] == ')':
#                    helper(s[:i] + s[i + 1:], i, lremove, rremove - 1);
                # 统计当前字符串中已有的括号数量

#        helper(s, 0, lremove, rremove)
#        return res


#分割等和子集
#思路先求nums的一半作为target，然后按照子集求target的思路来做
#行表示可以取nums[0:i]个备选，列表表示背包容量j，且每次判断的加入值为nums[i],j<nums[i]时，无法加入，i，j == i-1,j
#可以加入时有两种情况，取两种情况的或，有true即可
#初始化，第一行只有i==0,nums[i]为true
#第一列表示没有空间，那么所有物品都不选即可，所以都为true
#n = len(nums)
#if n < 2 or sum(nums) % 2 != 0:
#    return False
#else:
#    target = sum(nums) / 2
#total = sum(nums)
#maxNum = max(nums)
#if maxNum > target:
#    return False
#dp = [[False] * (target + 1) for _ in range(n)]
#for i in range(n):
#    dp[i][0] = True
#
#dp[0][nums[0]] = True
#for i in range(1, n):
#    num = nums[i]
#    for j in range(1, target + 1):
#        if j >= num:
#            dp[i][j] = dp[i - 1][j] | dp[i - 1][j - num]
#        else:
#            dp[i][j] = dp[i - 1][j]
#
#return dp[n - 1][target]

#路径求和三
#        def sumnode(root, target):
#            if not root:
#                return 0
#            res = 0
#            if root.val == target:
#                res += 1
#            res += sumnode(root.left, target - root.val)  # 更新target而不是累计求和
#            res += sumnode(root.right, target - root.val)
#            return res
#        if not root:
#            return 0
#        res = 0
#        queue = [root]
#        while queue:
#            tmp = queue.pop(0)
#            res += sumnode(tmp, targetSum)
#            if tmp.left:
#                queue.append(tmp.left)
#            if tmp.right:
#                queue.append(tmp.right)
#        return res
#前缀求法


#翻转链表
#        def reversenode(head):
#            pre, cur = None, head
#            while head:
#                next = cur.next
#                cur.next = pre
#                pre = cur
#                cur = next
#            return pre

##交错字符串的动规做法
#        if len(s1)+len(s2) != len(s3):
#            return False
#        len1=len(s1)
#        len2=len(s2)
#        len3=len(s3)
#        if(len1+len2!=len3):
#            return False
#        dp=[[False]*(len2+1) for i in range(len1+1)]
#        dp[0][0]=True
#        for i in range(1,len1+1):
#            dp[i][0]=(dp[i-1][0] and s1[i-1]==s3[i-1])
#        for i in range(1,len2+1):
#            dp[0][i]=(dp[0][i-1] and s2[i-1]==s3[i-1])
#        for i in range(1,len1+1):
#            for j in range(1,len2+1):
#                dp[i][j]=(dp[i][j-1] and s2[j-1]==s3[i+j-1]) or (dp[i-1][j] and s1[i-1]==s3[i+j-1])
#        return dp[-1][-1]

#单词接龙
#        def addWord(word):
#            if word not in wordId:
#                wordId[word] = self.nodeNum
#                self.nodeNum += 1

#        def addEdge(word):
#            addWord(word)
#            id1 = wordId[word]
#            chars = list(word)
#            for i in range(len(chars)):
#                tmp = chars[i]
#                chars[i] = "*"
#                newWord = "".join(chars)
#                addWord(newWord)
#                id2 = wordId[newWord]
#                edge[id1].append(id2)
#                edge[id2].append(id1)
#                chars[i] = tmp

#        wordId = dict()
#        edge = collections.defaultdict(list)
#        self.nodeNum = 0

#        for word in wordList:
#            addEdge(word)

#        addEdge(beginWord)
#        if endWord not in wordId:
#            return 0

#        dis = [float("inf")] * self.nodeNum
#        beginId, endId = wordId[beginWord], wordId[endWord]
#        dis[beginId] = 0

#        que = collections.deque([beginId])
#        while que:
#            x = que.popleft()
#            if x == endId:
#                return dis[endId] // 2 + 1
#            for it in edge[x]:
#                if dis[it] == float("inf"):
#                    dis[it] = dis[x] + 1
#                    que.append(it)
#        return 0