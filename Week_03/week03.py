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
[4]将子问题的结果进行合并，生成最终的结果

3.回溯，也是递归的思想，用一般的递归模板即可,需要remove current state

3.括号生成问题 使用一个栈来维护每次的状态，根据条件每次选择左括号还是右括号
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
def generateParenthesis(n):
    #递归
    stack = [('',0,0)]  #栈里面的元素分别表示当前的括号字符串，左括号的个数，右括号的个数
    res = []
    while stack:
        p,left,right = stack.pop()
        #终止条件
        if left == right == n:
            res.append(p)
            continue
        #每次只将符合条件的括号加入
        if left < n:
            stack.append((p + '(',left+1,right))
        if right < n and right < left:
            stack.append((p + ')',left,right+1))
    #栈为空的时候退出循环得到所有的括号
    return res

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

    # 二叉树的最近公共祖先
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        #递归终止条件 如果找到p或者是q,则直接返回
        if not root or root.val == p.val or root.val == q.val:
            return root

        #递归左右子树
        l = self.lowestCommonAncestor(root.left,p,q)
        r = self.lowestCommonAncestor(root.right,p,q)

        #分三种情况处理当前层逻辑
        #1.如果p,q既不再左子树，也不再右子树，则直接返回空
        if not l and not r:return None
        #2.p,q同时在左子树或右子树，则返回对应的左子树或右子树
        if not l:return r
        if not r:return l

        #3.p,q分别存在于左右子树当中，则直接返回当前节点作为最近的公共祖先
        return root

    # 从前序与中序遍历序列构造二叉树
    def buildTree(self,preorder, inorder):
        #递归实现
        #根据前序遍历找到根节点，然后找到中序遍历里面根节点的位置，进而可以知道左子树的长度
        #根据前序遍历左子树的边界和中序遍历左子树的边界构建左子树
        #根据前序遍历右子树的边界和中序遍历右子树的边界构建右子树
        n = len(preorder)
        hash_dict = {val:i for i,val in enumerate(inorder)}
        def helper(preorder_left,preorder_right,inorer_left,inorder_right):
            if preorder_left > preorder_right:
                return None
            #前序遍历第一个值即为根节点
            preorder_root = preorder_left
            #找到中序遍历对应的根节点的位置
            inorder_root = hash_dict[preorder[preorder_root]]
            #得到左子树的长度
            left_subtree_size = inorder_root - inorer_left
            #根据根节点构建树
            root = TreeNode(preorder[preorder_root])
            #构建左子树
            root.left = helper(preorder_left+1,preorder_left+left_subtree_size,inorer_left,inorder_root-1)
            #构建右子树
            root.right = helper(preorder_left+left_subtree_size+1,preorder_right,inorder_root+1,inorder_right)
            return root
        return helper(0,n-1,0,n-1)

#组合
def combine(n, k):
    #递归 和括号生成的问题道理一样 有K个位置 每个位置可以放对应的元素
    def helper(ans,tmp,n,k,begin):
        #递归终止条件
        if len(tmp) == k:
            ans.append(tmp)
            return

        #处理当前层的逻辑
        for i in range(begin,n+1):
            #将当前层的元素加入列表
            tmp.append(i)
            #递归下一层节点 如果不用tmp.copy() 则添加到ans里面的元素会随着tmp的变化而变化
            helper(ans,tmp.copy(),n,k,i + 1)
            #回到当前层，移除当前层的选择
            tmp.pop()
    ans = []
    helper(ans,[],n,k,1)
    return ans

#全排列
def permute(nums):
    #递归解决，分成多叉树
    def helper(ans,tmp,nums):
        #递归终止条件
        if len(tmp) == len(nums):
            ans.append(tmp)
            return

        #处理当前层逻辑
        for i in range(0,len(nums)):
            #排除已经选择过的
            if nums[i] in tmp:
                continue
            tmp.append(nums[i])
            helper(ans,tmp.copy(),nums)
            tmp.pop()
    ans = []
    helper(ans,[],nums)
    return ans

#全排列II
#Pow(x,n)
def myPow(x,n):
    #使用分治求解 自顶向下将大问题逐步分解成小的子问题
    def helper(n):
        #终止条件
        if n == 0:
            return 1
        #当前层的逻辑处理n //2 递归子问题
        y = helper(n // 2)
        #合并结果进行返回
        return y * y if n % 2 == 0 else y * y * x
    return helper(n) if n >= 0 else 1.0 / helper(-n)

#子集
def subsets(nums):
    #解法一：迭代遍历每个元素，将每个元素组个加入之前的子集中形成新的子集
    # [[]]
    # [[],[1]]
    # [[],[1],[2],[1,2]
    # [[],[1],[2],[3],[1,3],[2,3],[1,2,3]
    # res = [[]]
    # for i in nums:
    #     res = res + [[i] + num for num in res]
    # return res

    #解法二 递归 类似于括号生成的问题，对于每个数字，每次可以选或不选
    def helper(ans, nums, list, index):
        # 递归终止条件
        if index == len(nums):
            ans.append(list)
            return

            # 处理当前层，选或不选，递归子问题
        helper(ans, nums, list, index + 1)  # 不添加元素到list

        list.append(nums[index])  # 添加元素到List
        helper(ans, nums, list.copy(), index + 1)

        # 每个子问题处理之后需要将list面元素去掉
        list.pop()

    ans = []
    helper(ans, nums, [], 0)
    return ans


#多数元素
#电话号码的字母组合
def letterCombinations(digits):
        phoneMap = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }
        #创建递归函数
        def helper(i,digits,ans,tmp):
            """
            :param i: 表示第几个数字
            :param digits: 原始输入数字字符串
            :param ans: 结果列表
            :param tmp: 可能的字符列表
            :return:
            """
            #递归终止条件
            if i == len(digits):
                ans.append(''.join(tmp))
                return

            #处理当前层的逻辑
            for ch in phoneMap[digits[i]]:
                tmp.append(ch)
                #递归处理子问题
                helper(i + 1,digits,ans,tmp)
                tmp.pop()

        ans = []
        helper(0,digits,ans,[])
        return ans

#N皇后
def solveNQueens(n):
    cols = []
    pie = []
    na = []
    #递归每一行，记录可以存放的列
    def helper(ans,n,row,tmp):
        #递归终止条件
        if row >= n:
            ans.append(tmp)
            return

        #处理当前层的逻辑，即找出可以放皇后的列，并且将下一层不能放的位置都添加到对应的列表
        for col in range(n):
            if col in cols or row+col in pie or row-col in na:
                continue

            #否则,找到可以放的列
            cols.append(col)
            pie.append(row+col)
            na.append(row-col)

            #递归到下一行
            helper(ans,n,row+1,tmp.copy() + [col])

            #清除当前层的状态
            cols.pop()
            pie.pop()
            na.pop()

    ans = []
    helper(ans,n,0,[])

    result = []
    for item_lst in ans:
        sub_lst = []
        for i in item_lst:
            sub_lst.append('.'*i + 'Q' + '.'*(n-i-1))
        result.append(sub_lst)
    return result



if __name__ == '__main__':
    # print(generateParenthesis(3))
    # combinationSum([],[2,3,5],8)
    # print(combine(4,2))
    # print(letterCombinations('23'))
    # print(subsets([1,2,3]))
    print(permute([1,2,3]))

