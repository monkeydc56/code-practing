# while True:
#     try:
#         #内容
#     except:
#         break

#一行，一个数
#n = int(input())
#一行，多个数
# nn=list(map(int,input().split()))
# print(nn)
#两行，第一行一个数，第二行多个数
# n=int(input())
# nn=list(map(int,input().split())
#多行，但是没首行的数量索引
# import sys
# for line in sys.stdin:
#     nn = list(map(str,line.split()))#字符用str，数字用int，分别读每行，如果每行输出一个结果直接在for里面写就行，如果多行输出一个结果，就在存一下
#多行，一行是一个数，后面多个数
# nn = []
# n = int(input())
# for i in range(n):
#     a = list(map(int, input().split()))
#     nn.append(a)#可以直接接后面处理
#多行，第一行的第几个数是下面行的行数，下面每行有多个数
# mm=list(map(int,input().split()))
# nn = []
# for i in range(mm[2]):
#     a = list(map(int, input().split()))
#     nn.append(a)

# '//'除法取整
# '%'除法取余数

# abs(x)返回x的绝对值
# all(x)如果列表或者元组x所有元素不为0、’’、False或者iterable为空，all(iterable)返回True，否则返回False；注意：空元组、空列表返回值为True，这里要特别注意。
# any()函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True。
# 元素除了是 0、空、FALSE 外都算 TRUE。
# bin(x)返回x的二进制表示
# bool()函数用于将给定参数转换为布尔类型，如果没有参数，返回 False。
# bytearray()方法返回一个新字节数组。这个数组里的元素是可变的，并且每个元素的值范围: 0 <= x < 256。
# cmp(x,y)函数用于比较2个对象，如果 x < y 返回 -1, 如果 x == y 返回 0, 如果 x > y 返回 1。python3没有了
# compile()函数将一个字符串编译为字节代码。
# complex()函数用于创建一个值为 real + imag * j 的复数或者转化一个字符串或数为复数。如果第一个参数为字符串，则不需要指定第二个参数。
# dict()用于创建字典。
# divmod(x,y)返回x/y的(商，余数)
# eval()用来执行一个字符串表达式，并返回表达式的值eval(3+4)=7
# float()函数用于将整数和字符串转换成浮点数
# frozenset()返回一个冻结的集合，冻结后集合不能再添加或删除任何元素
# hash()用于获取取一个对象（字符串或者数值等）的哈希值
# help()函数用于查看函数或模块用途的详细说明
# hex()函数用于将10进制整数转换成16进制，以字符串形式表示
# input() 函数接受一个标准输入数据，返回为 string 类型
# int()函数用于将一个字符串或数字转换为整型
# len()对象（字符、列表、元组等）长度或项目个数
# list()将元组转换为列表
# long()函数将数字或字符串转换为一个长整型
# max()返回最大值
# min()返回最小值
# oct()函数将一个整数转换成 8 进制字符串
# pow(x,y)计算x的y次方
# print()打印
# range()函数可创建一个整数列表
# reduce(x,[list])用x计算规则对list进行累计（from functools import reduce）
# reverse(list)函数用于反向列表中元素
# round(x)方法返回浮点数x的四舍五入值
# set(list)函数创建一个无序不重复元素集
# list[slice(x，y，z)]截取列表x为开始位置，y为结束位置，z为间距
# sorted(list，reverse=False/True)对列表排序，F为升序，T为降序
# str()转出字符串
# sum()求和
# tuple()函数将列表转换为元组
# vars()函数返回对象object的属性和属性值的字典对象。
# zip(list1,list2)将两个列表的对应位组合成新列表[(list1[0],list2[0]),…],元素个数与最短的列表一致

# 字符串
# str.lower()全部小写
# str.upper()全部大写
# str.capitalize()首字母大写
# str.title()每个词的首字母大写
# str.swapcase()所有词大小写互换
# list(str)，将字符串中的所有元素逐个加入list中包括字母、符合、空格
#
# 列表
# 列表list=[1，2，3]
# list*2输出两次列表[1，2，，3，1，2，3],+可以把两个列表左右拼起来（但是不能减）.
# list.append(x)在列表最后添加x
# list.insert(x,y)在列表第x位添加y
# list.remove(x)删除列表的第一个x
# list.index(x)列队的第一个x在第几位
# list.count(x)列表中x出现的次数
# list.sort(reverse=False)列表升序，True降序排列
# max(list)列表最大值
# min(list)列表最小值
# len(list)列表中元素个数
# sum(list)列表中所有元素的和
# list.extend(list2)相当于list+list2
# list.pop(x)输出列表x位置的元素,a=list.pop(x)，a为删除的那个值
# list.pop()删除列表最后一位的元素
# list.reverse()翻转列表中的所有元素
#
# 集合
# set.add(x)字典添加x
# set.clear字典清空
# set.remove(x)删除字典中的x
# set.discard(x)删除字典中的x
# set3 = set1.different(set2)找set1和set2不同的数
# set3 = set1.interection(set2)找set1和set2相同的数
# print(a - b) # a 和 b 的差集
# print(a | b) # a 和 b 的并集
# print(a & b) # a 和 b 的交集
# print(a ^ b) # a 和 b 中不同时存在的元素
#
# 字典{}
# {key1:value1,key2:value2}同一个字典key是唯一的，
# dictionary.keys（）所有的key
# dictionary.values（）所有的value
# 构建字典时可以用dict（key=value，[（‘key’，value）]）

# 生成函数：
# np.array(list)将输入数据转化为一个ndarray
# np.ones(n)生成一个长度为n的，全是1的ndarray
# np.zeros(n)生成一个长度为n的，全是0的ndarray
# np.eye(n)创建一个n*n的单位矩阵（对角线为1，其余为0）
# np.arange(num)生成一个从0到num-1步数为1的一维ndarray类似(range)
# np.where(cond, ndarray1, ndarray2)根据条件cond，选取ndarray1或者ndarray2，返回一个新的ndarray。满足条件的输出ndarray1，不满足的输出ndarray2。
# np.in1d(ndarray, [x,y,…])检查ndarray中的元素是否等于[x,y,…]中的一个，有返回True，无返回False。
# np.linspace(x,y,z)将x到y的均分成z-1段，包含xy共5个点
# x[:, np.newaxis]把x作为列增加一个维度（相当于只有一列的矩阵）
# x[np.newaxis, :]把x作为行增加一个维度（相当一把一个列表变成一个只有一行的矩阵）
#
# 矩阵函数：
# np.diag( ndarray)以一维数组的形式返回方阵的对角线（或非对角线）元素
# np.diag( [x,y,…])将一维数组转化为方阵（非对角线元素为0）
# np.dot(ndarray, ndarray)矩阵乘法
# np.trace(ndarray)计算对角线元素的和
# np.linalg.inv()求逆矩阵
# np.linalg中的函数solve可以求解形如 Ax = b 的线性方程组，其中 A 为矩阵，b 为一维或二维的数组，x 是未知变量
# c1,c2=np.linalg.eig()模块中，eigvals函数可以计算矩阵的特征值，而eig函数可以返回一个包含特征值和对应的特征向量的元组。c1是特征值，c2是特征向量。
# np.linalg.det()求矩阵的行列式
# np.vstack((a,b))a,b上下合并
# np.hstack((a,b))a,b左右合并
# np.concatenate((a,b,c),axis=0)axis=0时abc上下合并，axis=1时abc左右合并
# np.vsplit(a,3)a切成上下3块
# np.hsplit(a,3)a切成左右3块
# np.split(d,2,axis=1)axis=1把矩阵d分成左右两部分，axis=0把矩阵d分为上下两部分
# np.array_split(a,3,axis=1)不均衡分类，其他规则同上
#
# 排序函数：
# np.sort(ndarray)排序
# np.unique(ndarray)返回ndarray中的元素，排除重复元素之后，并进行排序
# np.intersect1d( ndarray1, ndarray2)返回二者的交集并排序
# np.union1d( ndarray1, ndarray2)返回二者的并集并排序
# np.setdiff1d( ndarray1, ndarray2)返回二者的差
# np.setxor1d( ndarray1, ndarray2)返回二者的对称差
#
# 一元计算函数
# np.abs(ndarray)绝对值
# np.mean(ndarray)平均值
# np.sqrt(ndarray)计算x^0.5
# np.square(ndarray)计算x^2
# np.exp(ndarray)计算e^x
# np.log、log10、log2
# np.sign(ndarray)	计算正负号：1（正）、0（0）、-1（负）
# np.ceil(ndarray)进一
# np.floor(ndarray)去尾
# np.rint(ndarray)四舍五入
# np.modf(ndarray)将数组的小数和整数部分以两个独立的数组方式返回
# np.cos、cosh、sin、sinh、tan、tanh、arccos、arccosh、arcsin、arcsinh、arctan、arctanh三角函数
#
# 多元计算函数：
# np.add(ndarray, ndarray)相加
# np.subtract(ndarray, ndarray)相减
# np.multiply(ndarray, ndarray)乘法
# np.divide(ndarray, ndarray)除法
# np.floor_divide(ndarray, ndarray)圆整除法（丢弃余数）
# np.power(x, y)x的y次方
# np.mod(x, y)求模
#
# 函数：
# ndarray.reshape((N,M,…))将ndarray转化为NM…的多维ndarray（非copy）
#
# 计算函数：
# ndarray.mean( axis=0 )求平均值
# ndarray.sum( axis= 0)求和
# ndarray.cumsum( axis=0)累加
# ndarray.cumprod( axis=0)累乘
# ndarray.std()方差
# ndarray.var()标准差
# ndarray.max()最大值
# ndarray.min()最小值
# ndarray.argmax()最大值索引
# ndarray.argmin()最小值索引
# ndarray.median()中位数