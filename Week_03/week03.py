# -*- encoding:utf-8 -*-
"""
递归，树的递归
分治回溯
"""

"""
1.递归代码模板
[1]递归终止条件
[2]处理当前层逻辑
[3]递归到下一层
关键：找最近重复子问题，数学归纳法

2.分治代码模板
关键：找问题的重复性
[1]递归终止条件
[2]处理当前逻辑（处理当前问题，分解子问题）
[3]递归处理子问题
[4]生成最终的结果

3.回溯，也是递归的思想，用一般的递归模板即可

3.括号生成问题
"""
#爬楼梯
def climbStairs(n):
    #思路：动态规划
    if n <= 3:
        return n
    f1,f2,f3 = 1,2,3
    for i in range(3,n+1):
        f3 = f1 + f2
        f1 = f2
        f2 = f3
    return f3

#括号生成

#二叉树的最大深度
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    #二叉树的最大深度
    def maxDepth(self, root: TreeNode) -> int:
        #解法一：递归，最大深度等于左子树和右子树的最大深度加1
        if not root:
            return 0
        left_height = self.maxDepth(root.left)
        right_height = self.maxDepth(root.right)
        return max(left_height,right_height) + 1

    #二叉树的最小深度
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0

        # 递归终止条件分三种情况
        # 1.左右子树都为空
        if (not root.right) and (not root.left):
            return 1

        left_height = self.minDepth(root.left)
        right_height = self.minDepth(root.right)

        # 2.左子树和右子树有一个为空的情况
        if (not root.right) or (not root.left):
            return left_height + right_height + 1

        # 左右子树都不为空的情况
        return min(left_height, right_height) + 1

    #翻转二叉树
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return root
        #左子树翻转后的结果
        left = self.invertTree(root.left)
        right = self.invertTree(root.right)
        root.left = right
        root.right = left
        return root

    #验证二叉搜索树
    def isValidBST(self, root: TreeNode) -> bool:
        #解法一：递归判断每个节点的值是否在正确的范围内
        # def helper(root,left,right):
        #     if not root:
        #         return True
        #
        #     if (root.val > left) and (root.val < right):
        #         return helper(root.left,left,root.val) and helper(root.right,root.val,right)
        #     else:
        #         return False
        # return helper(root,-float('inf'),float('inf'))

        #解法二：采用栈实现中序遍历，判断出栈当前节点的值是否比前一个值大
        cur,stack,res = root,[],[]
        inorder = -float('inf')
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.left
            tmp = stack.pop()
            if tmp.val < inorder:
                return False
            inorder = tmp.val
            cur = tmp.right
        return True


#二叉树的序列化与反序列化
#二叉树的最近公共祖先
#从前序与中序遍历序列构造二叉树
#组合
#全排列
#全排列II
#Pow(x,n)
def myPow(x,n):
    #使用分治求解
    def helper(n):
        if n == 0:
            return 1
        y = helper(n // 2)
        return y * y if n % 2 == 0 else y * y * x
    return helper(n) if n >= 0 else 1.0 / helper(-n)

#子集
def subsets(nums):
    #解法一：迭代遍历每个元素，将每个元素组个加入之前的子集中形成新的子集
    # [[]]
    # [[],[1]]
    # [[],[1],[2],[1,2]
    # [[],[1],[2],[3],[1,3],[2,3],[1,2,3]
    res = [[]]
    for i in nums:
        res = res + [[i] + num for num in res]
    return res

#多数元素
#电话号码的字母组合
#N皇后

