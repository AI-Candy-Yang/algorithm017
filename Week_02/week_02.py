# -*- encoding:utf-8 -*-
import collections
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
#本周练习
#1.哈希表，映射，集合
#2.树，二叉树
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children

    #3.N叉树的后序遍历  左右根
    def postorder(self, root: 'Node'):
        #解法一：递归解法 从左往右分别加入子节点的值，最后加入根节点的值
        # if not root:
        #     return []
        # res = []
        # for child in root.children:
        #     res.extend(self.postorder(child))
        # res.append(root.val)
        # return res

        #解法二：迭代解法 使用栈，前序遍历为根左右  入栈的时候从左到右，则出栈即为根右左，将顺序进行逆序即为左右根
        if not root:
            return []
        res,stack = [],[root]
        while stack:
            cur = stack.pop()
            res.append(cur.val)
            stack.extend(cur.children)
        return res[::-1]

    # 3.N叉树的前序遍历
    def preorder(self, root: 'Node'):
        # 解法一：递归 根左右的方式加入结果列表
        # if not root:
        #     return []
        # res = [root.val]
        # for child in root.children:
        #     res.extend(child.val)
        # return res

        # 解法二：迭代 使用栈 将子节点采用从右到左的方向入栈  则出栈为左右
        if not root:
            return []
        res, stack = [], [root]
        while stack:
            cur = stack.pop()
            res.append(cur.val)
            stack.extend(cur.children[::-1])
        return res

    #7.N叉树的层序遍历
    def levelOrder(self, root: 'Node'):
        #使用队列进行
        if not root:
            return []
        res = []
        queue = collections.deque([root])
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                res.append(node.val)
                queue.extend(node.children)
            res.append(level)
        return res

#1.最小的k个数
import heapq
def getLeastNumbers(arr, k):
    #通过堆来实现 要求的是最小的K个值，那么需要每次将较大的元素替换出去，因此需要维护一个大根堆
    #python 默认的是小根堆  因此需要取相反数
    #每个元素插入删除的时间复杂度都是O(logk)  总的时间复杂度为O(nlogk)
    if k == 0:
        return []
    #取数组里面前k个元素的相反数
    arr = [-x for x in arr[:k]]
    #构建最小堆
    hp = heapq.heapify(arr)
    for i in range(k,len(arr)):
        if -hp[0] > arr[i]:
            heapq.heappop(hp)
            heapq.heappush(hp,-arr[i])
    #最后取出堆上的元素
    res = [-x for x in hp]
    return res

#2.滑动窗口最大值


#本周作业
#1.有效的字母异位词（每个字母出现的次数是一样的，但是位置不一样）
def isAnagram(s, t):
    #解法一：直接通过字符串排序进行判断  时间复杂度为O（nlogn)
    # s_sorted = ''.join(sorted(s))
    # t_sorted = ''.join(sorted(t))
    # if s_sorted == t_sorted:
    #     return True
    # else:
    #     return False

    #解法二：直接统计每个字符出现的次数
    # return collections.Counter(s) == collections.Counter(t)

    #解法三：使用哈希表统计每个字符串里面字符出现的次数
    if len(s) != len(t):
        return False
    counter = {}
    for char in s:
        if char in counter:
            counter[char] += 1
        else:
            counter[char] = 1

    for char1 in t:
        if char1 in counter:
            counter[char1] -= 1
        else:
            return False

    for value in counter.values():
        if value != 0:
            return False
    return True

#2.两数之和
def twoSum(nums, target):
    #思路：采用哈希表，哈希表查找元素的时间复杂度为O(1)
    hash_dict = {}
    for i,num in enumerate(nums):
        if (target - num) in hash_dict:
            return [hash_dict[target-num],i]
        hash_dict[num] = i

#4.字母异位词分组
def groupAnagrams(strs):
    #解法一：对字符串排序后分组  对字符串排序后转成tuple得到不可变对象作为字典的键
    # res_dict = {}
    # for s in strs:
    #     res_dict[tuple(sorted(s))].append(s)
    # return res_dict.values()

    #解法二：通过将字符转为0-26之间的数字，然后记录每个数字出现的次数序列作为键
    res_dict = {}
    for s in strs:
        count = [0] * 26
        #遍历每个字符，统计每个字符出现的次数
        for c in s:
            count[ord(c) - ord('a')] += 1
        #将该字符串出现的次数序列转化位tuple作为键
        res_dict[tuple(count)].append(s)
    return res_dict.values()


#5.二叉树的中序遍历
class Solution:
    def inorderTraversal(self, root: TreeNode):
        #解法一：递归解决 时间复杂度为O(n) n为节点个数   空间复杂度为O(h) h为树的深度
        # if not root:
        #     return []
        # return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

        #解法二：迭代 使用栈先进后出，中序遍历需要先找到最左边的叶子节点，然后采用左中右的方式进行出栈，指针指向当前出栈节点的右子节点
        if not root:
            return []

        #初始化当前指针，结果列表和栈
        cur,res,stack = root,[],[]
        while cur or stack:
            #找到最左边的叶子节点
            while cur:
                #依次将遇到的节点加入栈
                stack.append(cur)
                #指向下一个左子节点
                cur = cur.left
            #找到最左边的叶子节点后开始进行出栈操作,并记录出栈的节点
            tmp = stack.pop()
            #将栈顶元素加入结果列表
            res.append(tmp.val)
            #将当前指针指向栈顶节点的右子节点
            cur = tmp.right
        return res


    #6.二叉树的前序遍历
    def preorderTraversal(self, root: TreeNode):
        #解法一：递归解法
        # if not root:
        #     return []
        # return [root.val] + preorderTraversal(root.left) + preorderTraversal(root.right)

        #解法二：迭代解法 加入根节点的值，然后依次将右子节点和左子节点入栈
        if not root:
            return []

        res,stack = [],[root]
        while stack:
            #获取栈顶元素
            cur = stack.pop()
            res.append(cur.val)
            #依次将右子节点和左子节点入栈
            if cur.right:
                stack.append(cur.right)
            if cur.left:
                stack.append(cur.left)
        return res

#8.丑数
#9.前K个高频元素
