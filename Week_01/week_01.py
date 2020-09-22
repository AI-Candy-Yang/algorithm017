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


if __name__ =='__main__':
    moveZeroes([0,1,0,3,12])