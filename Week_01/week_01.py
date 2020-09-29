#-*- encoding:utf-8 -*-

#作业
#1.删除排序数组中的重复项,返回去重后的数组长度
def removeDuplicates(nums):
    # 双指针 快指针指向不重复的元素
    low = 0
    for fast in range(len(nums)):
        if (nums[fast] != nums[low]):
            low += 1
            nums[low] = nums[fast]
    return low + 1

#2.旋转数组
def rotate(nums, k):
    """
    Do not return anything, modify nums in-place instead.
    """
    #解法一：将数组切分成两部分，前面的n-k和后面的k个元素
    #首先将两个子数组进行反转[4,3,2,1,7,6,5]
    #然后将整体进行反转 [5,6,7,1,2,3,4]
    # def reverse(l,r):
    #     while l < r:
    #         nums[l],nums[r] = nums[r],nums[l]
    #         l += 1
    #         r -= 1
    # k = k % len(nums)
    # reverse(0,len(nums)-k-1)
    # reverse(len(nums)-k,len(nums)-1)
    # reverse(0,len(nums)-1)

    #解法二：使用额外的数组
    n = len(nums)
    a = [0] * n
    for i in range(n):
        a[(i+k)%n] = nums[i]

    for i in range(n):
        nums[i] = a[i]


#3.合并两个有序链表
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    #递归
    # if not l1:
    #     return l2
    # if not l2:
    #     return l1
    # if l1.val < l2.val:
    #     l1.next = mergeTwoLists(l1.next,l2)
    #     return l1
    # else:
    #     l2.next = mergeTwoLists(l1,l2.next)
    #     return l2

    #迭代
    prehead = ListNode(-1)
    pre = prehead
    while l1 and l2:
        if l1.val < l2.val:
            pre.next = l1
            l1 = l1.next
        else:
            pre.next = l2
            l2 = l2.next
        pre = pre.next

    #将未被合并的直接添加到末尾
    pre.next = l1 if l1 is not None else l2
    return prehead.next


#4.合并两个有序列表
def merge(nums1, m, nums2, n):
    #解法一，从前往后开始比较，申请一个新的数组空间，时间复杂度为O(n+m)  空间复杂度为O(m)
    #获取nums1有效的元素
    nums1_copy = nums1[:m]
    #将nums1数组赋值为空数组
    nums1[:] = []
    while (len(nums1_copy) > 0) and (len(nums2) > 0):
        if (nums1_copy[0] < nums2[0]):
            nums1.append(nums1_copy.pop(0))
        else:
            nums1.append(nums2.pop(0))
    #将两个数组未比较完的元素添加进来
    nums1.extend(nums1_copy)
    nums1.extend(nums2)

    #解法二：从后往前开始比较，不需要申请额外的内存空间  时间复杂度为O(m+n)  空间复杂度为O(1)
    p1 = m - 1
    p2 = n - 1
    p = m + n - 1
    while (p1 > 0) and (p2 > 0):
        if nums1[p1] < nums2[p2]:
            nums1[p] = nums2[p2]
            p2 -= 1
        else:
            nums1[p] = nums1[p1]
            p1 -= 1
        p -= 1

    #将最后多的元素加到前面，只可能有一个数组不为空，如果是第一个不为空的话，num1不需要改变
    nums1[:p2+1] = nums2[:p2+1]

#5.两数之和
def twoSum(nums, target):
    if len(nums) < 2:
        return []

    n = len(nums)
    # 排序加双指针
    nums.sort()
    l = 0
    r = n - 1
    while l < r:
        sum1 = nums[l] + nums[r]
        if sum1 < target:
            l += 1
        elif sum1 > target:
            r -= 1
        else:
            return [l,r]

#6.移动零
def moveZeroes(nums):
    #解法一：快慢指针解法，快指针用于查找非零元素，慢指针用于记录非零元素存放的位置
    # low = 0
    # for fast in range(len(nums)):
    #     if (nums[fast] != 0):
    #         nums[low] = nums[fast]
    #         if (fast != low):
    #             nums[fast] = 0
    #         low += 1


    #解法二:当快指针指向非零元素的时候直接蒋欢快慢指针的位置，然后同步前进
    low = 0
    for fast in range(len(nums)):
        if (nums[fast] != 0):
            nums[low],nums[fast] = nums[fast],nums[low]
            low += 1

    print(nums)

#7.加一
def plusOne(digits):
    n = len(digits)
    for i in range(n-1,-1,-1):
        digits[i] += 1
        digits[i] %= 10
        #如果不等于0，则说明这个位置原来不为9，直接返回即可，如果等于0，则直接移到前一位加1，继续判断
        if digits[i] != 0:
            return digits
    #如果都没有返回，表示所有的位置加1后都变成了0，则直接在首尾加1
    digits.insert(0, 1)
    return digits

#8.设计循环双端队列

#9.接雨水
def trap(height):
    #和柱状图中最大的矩形比较像 遍历柱子的高度，找出左右边界
    #解法一：对每个元素，找出下雨后谁能达到的最大高度（左右两边最大高度的最小值减去当前高度值），最后将所有的高度相加
    #时间复杂度为O(n^2)  空间复杂度为O(1)
    # res = 0
    # for i in range(len(height)):
    #     max_left = 0
    #     max_right = 0
    #     #分别从当前元素向左和向右查找
    #     for j in range(i,-1,-1):
    #         max_left = max(max_left,height[j])
    #
    #     #向右遍历
    #     for j in range(i,len(height)):
    #         max_right = max(max_right,height[j])
    #
    #     #加上每个元素可以接雨水的高度
    #     res += min(max_left,max_right) - height[i]
    # return res

    #解法二：提前存储每个位置可以看到的左边和右边的最大值
    # if len(height) <= 1:
    #     return 0
    #
    # res = 0
    # n = len(height)
    # left_max = [0]*n
    # right_max = [0]*n
    #
    #
    # #从左往右遍历记录每个位置的最大值
    # left_max[0] = height[0]
    # for i in range(n):
    #     left_max[i] = max(height[i],left_max[i-1])
    #
    # #从右往左记录每个位置的最大值
    # right_max[n-1] = height[n-1]
    # for i in range(n-2,-1,-1):
    #     right_max[i] = max(height[i],right_max[i+1])
    #
    # #结合每个位置的左边的最大值和右边的最大值计算
    # for i in range(n):
    #     res += min(left_max[i],right_max[i]) - height[i]
    #
    # return res

    #解法三：按照柱状图中最大的矩形面积使用单调栈来找出每个位置的左右边界，栈底到栈顶由大变小，维持一个单调递减的栈
    #当前元素比栈顶元素大 栈顶元素出栈
    #当前元素比栈顶元素小 则继续入栈
    n = len(height)
    if n < 3:
        return 0
    res,idx = 0,0
    stack = []
    while idx < n:
        while len(stack) > 0 and height[idx] > height[stack[-1]]:
            top = stack.pop()
            if len(stack) == 0:
                break
            #高度为左边界高度和右边界高度最小值-当前元素高度
            h = min(height[stack[-1]],height[idx])-height[top]
            #间距
            dist = idx - stack[-1] - 1
            res += (dist * h)
        stack.append(idx)
        idx += 1
    return res

#练习
#2.盛水的最大容器面积
#双指针左右移动找出最大的面积，最大面积纸盒左右两个柱子的高度有关
def maxArea(heights):
    l,r = 0,len(heights)-1
    area = 0
    while (l < r):
        if heights[l] < heights[r]:
            area = max(area,(r-l)*heights(l))
            l += 1
        else:
            area = max(area,(r-l)*heights[r])
            r -= 1
    return area

#3.爬楼梯
def climbStairs(n):
    if n <= 3:
        return n
    x,y,z = 1,2,3
    for i in range(4,n+1):
        x,y,z = y,z,y+z
    return z

#4.三数之和
#从头到尾遍历表示其中一个数
def threeSum(nums):
    #排序+遍历+双指针
    if len(nums) < 3:
        return []
    res = []
    for i in range(len(nums)-2):
        if (i > 0) and (nums[i] == nums[i-1]):
            continue
        l = i + 1
        r = len(nums) - 1
        while (l < r):
            sum1 = nums[i] + nums[l] + nums[r]
            if sum1 == 0:
                res.append([nums[i],nums[l],nums[r]])
                while (l < r) and (nums[l] == nums[l+1]):
                    l += 1
                while (l < r) and (nums[r] == nums[r-1]):
                    r -= 1
                l += 1
                r -= 1
            elif sum1 > 0:
                r -= 1
            else:
                l -= 1
    return res

#5.反转链表
def reverseList(self, head: ListNode) -> ListNode:
    #初始化前一个节点
    pre = None
    #初始化当前节点
    cur = head
    while cur:
        #初始化临时变量保存当前节点的下一节点
        tmp = cur.next
        #将当前节点指向前一个节点
        cur.next = pre
        #向后移动前一个节点和当前节点
        pre = cur
        cur = tmp
    #返回头节点 pre
    return pre

#6.两两交换链表中的节点
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if (not head) or (not head.next):
            return head
        #初始化下个节点
        next = head.next
        #将当前节点指向下下个节点进行交换后的子链表
        head.next = self.swapPairs(next.next)
        #下个节点指向当前节点
        next.next = head
        return next

#7.环形链表
def hasCycle(self, head: ListNode) -> bool:
    #设置快慢指针 快指针每次走两步，慢指针每次走一步，如果快慢指针相遇，则表示有环
    if (not head) or (not head.next):
        return False

    low = fast = head
    while fast and fast.next:
        low = low.next
        fast = fast.next.next
        if low is fast:
            return True
    else:
        return False

#8.K个一组翻转链表
#9.有效的括号
def isValid(s):
    #使用栈，左括号入栈，判断和栈顶元素能否抵消
    if len(s) % 2 != 0:
        return False

    pairs = {"]":"[",")":"(","}":"{"}
    stack = []
    for ch in s:
        #如果当前括号是右括号
        if ch in pairs.keys():
            #栈顶元素和当前括号不匹配
            if not stack or stack[-1] != pairs[ch]:
                return False
            #匹配的情况，直接将栈顶元素出栈
            stack.pop()
        else: #当前括号是左括号，则入栈
            stack.append(ch)
    return not stack


#10.柱状图中最大的矩形面积
def largestRectangleArea(heights):
    #解法一 遍历高度，分别找出每根柱子的左右边界
    # res = 0
    # n = len(heights)
    # for i in range(n):
    #     left_i = i
    #     right_i = i
    #     while left_i > 0 and heights[i] <= heights[left_i]:
    #         left_i -= 1
    #     while right_i < n and heights[i] <= heights[right_i]:
    #         right_i += 1
    #     res = max(res,(right_i-left_i-i)*heights[i])
    # return res

    #解法二：使用递增的栈来获取每个元素的左右边界
    heights = [0] + heights + [0]
    res = 0
    stack = []
    for i in range(len(heights)):
        #栈里面存放的是每个元素的索引
        while stack and heights[stack[-1]] > heights[i]:
            #记录栈顶元素的索引
            tmp = stack.pop()
            #根据左右边界计算对应的面积
            res = max(res,(i-stack[-1]-1)*tmp)
        #如果大于栈顶元素则直接入栈
        stack.append(i)
    return res

import collections
#11.滑动窗口最大值
def maxSlidingWindow(nums,k):
    #采用双端队列
    if len(nums) < 2:
        return nums
    queue = collections.deque()
    res = []
    for i in range(len(nums)):
        #将元素加入双端队列，保证从大到小排序，新加入的如果比队尾元素大，则删除队尾元素，加入新得元素
        while queue and nums[queue[-1]] < nums[i]:
            queue.pop()

        #当队列为空，或加入的元素比队尾元素小，则直接加入队列
        queue.append(i)

        #判断队首是否在窗口内部
        if queue[0] <= i - k:
            queue.popleft()

        #当窗口长度为k时加入队首元素到结果列表
        if i + 1 >= k:
            res.append(nums[queue[0]])

        return res






if __name__ =='__main__':
    # moveZeroes([0,1,0,3,12])
    print(twoSum([2, 7, 11, 15],9))