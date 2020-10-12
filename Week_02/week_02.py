# -*- encoding:utf-8 -*-
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
#本周练习
#1.哈希表，映射，集合
#2.树，二叉树
#1.二叉树的中序遍历
def inorderTraversal(self, root: TreeNode):
    #迭代算法  先找到最左边的子节点 ，并且经过的节点依次入栈 找到后依次进行出栈，并且将出栈元素加到结果列表，指向右节点
    if not root:
        return []
    cur,res,stack = root,[],[]
    while stack or cur:
        #找到最左边的节点
        while cur:
            #将节点依次入栈
            stack.append(cur)
            cur = cur.left
        #开始出栈
        tmp = stack.pop()
        #将出栈的元素加入结果列表
        res.append(tmp.val)
        #指针指向右节点
        cur = tmp.right
    return res

#2.二叉树的前序遍历
def preorderTraversal(self, root: TreeNode):
    #迭代解法 根左右的顺序出栈 依次将根节点-右子节点-左子节点入栈
    if not root:
        return []

    res,stack = [],[root]
    while stack:
        cur = stack.pop()
        res.append(cur.val)
        if cur.right:
            stack.append(cur.right)
        if cur.left:
            stack.append(cur.left)
    return res

class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children

    #3.N叉树的后序遍历
    def postorder(self, root: 'Node'):
        #解法一：递归解法
        # if not root:
        #     return []
        # res = []
        # for child in root.children:
        #     res.extend(self.postorder(child))
        # res.append(root.val)
        # return res

        #解法二：迭代解法
        if not root:
            return []
        res,stack = [],[root]
        while stack:
            cur = stack.pop()
            res.append(cur.val)
            stack.extend(cur.children)
        return res[::-1]

#4.N叉树的前序遍历
def preorder(self, root: 'Node'):
    #解法一：递归
    # if not root:
    #     return []
    # res = [root.val]
    # for child in root.children:
    #     res.extend(child.val)
    # return res

    #解法二：迭代
    if not root:
        return []
    res,stack = [],[root]
    while stack:
        cur = stack.pop()
        res.append(cur.val)
        stack.extend(cur.children[::-1])
    return res


#3.堆，二叉堆
#1.最小的k个数
#2.滑动窗口最大值


#本周作业
#1.有效的字母异位词（每个字母出现的次数是一样的，但是位置不一样）

#2.两数之和
#3.N叉树的前序遍历
def preorder(self, root: 'Node'):
    #解法一：递归
    # if not root:
    #     return []
    # res = [root.val]
    # for child in root.children:
    #     res.extend(child.val)
    # return res

    #解法二：迭代
    if not root:
        return []
    res,stack = [],[root]
    while stack:
        cur = stack.pop()
        res.append(cur.val)
        stack.extend(cur.children[::-1])
    return res
#4.字母异位词分组
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


#7.N叉树的层序遍历
import collections
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

#8.丑数
#9.前K个高频元素
