#-*- encoding:utf-8 -*-

#1.移动零
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

#两数之和
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


if __name__ =='__main__':
    # moveZeroes([0,1,0,3,12])
    print(twoSum([2, 7, 11, 15],9))