# nums = [[i for i in range(500)] for i in range(20)]
# index, res = 0, []
# while index<500:
#     candates = nums[0][0]
#     flag = 0
#     while i < 20:
#         tmp = nums[i][0]
#         if tmp >= candates:
#             candates = tmp
#             flag = i
#         i += 1
#     nums[flag] = nums[flag][1:]
#     res.append(candates)
#     index +=1
#     i = 0
# print(res)
import sys
# def minstep(s):
#     n=len(s)
#     k=0
#     sum=0
#     for i in range(n):
#         sum+=1
#         if k==0:
#             if s[i]>='A' and s[i]<='Z':
#                 sum+=1
#                 if i<n-1 and s[i+1]>='A' and s[i+1]<='Z':
#                     k=1
#         if k==1:
#             if s[i]>='a' and s[i]<='z':
#                 sum+=1
#                 if i<n-1 and s[i+1]>='a' and s[i+1]<='z':
#                     k=0
#     return sum
# index = 0
# while True:
#     line = sys.stdin.readline().strip()
#     if line.isdigit():
#         continue
#     if line == "":
#         break
#     result = minstep(line)
#     print(result)
