package code

import (
	"fmt"
	"sort"
	"strings"
)

// Implement pow(x, n), which calculates x raised to the power n (i.e., xn)

func myPow(x float64, n int) float64 {
	answer := x
	if n < 0 {
		c := n * -1
		for i := 2; i <= c; i++ {
			answer *= x
		}
		return (1 / answer)
	} else if n == 0 {
		return answer
	} else {
		for i := 2; i <= n; i++ {
			answer *= x
		}
		return answer
	}

}

// Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

// You may assume that each input would have exactly one solution, and you may not use the same element twice.

// You can return the answer in any order.

func twoSum(nums []int, target int) []int {
	for i := 0; i < len(nums); i++ {
		for j := 0; j < len(nums); j++ {
			if i == j {
				continue
			}
			if nums[i]+nums[j] == target {
				return []int{i, j}
			}
		}
	}
	return []int{}
}

// Given a string s, find the length of the longest
// substring
//  without repeating characters.

func lengthOfLongestSubstring(s string) int {
	max := 0
	for i := 0; i < len(s); i++ {
		tmpByte := make(map[byte]bool)
		for j := i; j < len(s); j++ {
			if tmpByte[s[j]] {
				break
			}
			tmpByte[s[j]] = true
		}
		if len(tmpByte) > max {
			max = len(tmpByte)
		}
	}

	return max
}

// You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

// You may assume the two numbers do not contain any leading zero, except the number 0 itself.

type ListNode struct {
	Val  int
	Next *ListNode
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	result := &ListNode{}
	now := result
	overhead := 0

	for l1 != nil || l2 != nil {
		first := 0
		second := 0
		if l1 != nil {
			first = l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			second = l2.Val
			l2 = l2.Next
		}
		sum := first + second + overhead
		overhead = sum / 10
		now.Next = &ListNode{Val: sum % 10}
		now = now.Next
	}

	if overhead > 0 {
		now.Next = &ListNode{Val: overhead}
	}

	return result.Next
}

// Given an integer x, return true if x is a palindrome, and false otherwise.

func isPalindrome(x int) bool {
	if x < 0 || (x != 0 && x%10 == 0) {
		return false
	}
	rev := 0
	for x > rev {
		rev = rev*10 + x%10
		x = x / 10
	}
	return (x == rev) || (x == rev/10)
}

// Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

// Symbol       Value
// I             1
// V             5
// X             10
// L             50
// C             100
// D             500
// M             1000
// For example, 2 is written as II in Roman numeral, just two one's added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

// Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

// I can be placed before V (5) and X (10) to make 4 and 9.
// X can be placed before L (50) and C (100) to make 40 and 90.
// C can be placed before D (500) and M (1000) to make 400 and 900.
// Given an integer, convert it to a roman numeral.

func intToRoman(num int) string {
	symbol := []string{"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"}
	value := []int{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1}

	result := ""

	for num > 0 {
		for i := range value {
			if num >= value[i] {
				result += symbol[i]
				num -= value[i]
				break
			}
		}
	}
	return result

}

//Roman to int

func romanToInt(s string) int {
	mapping := map[rune]int{
		'I': 1,
		'V': 5,
		'X': 10,
		'L': 50,
		'C': 100,
		'D': 500,
		'M': 1000,
	}
	var sum int
	var prev rune
	for _, val := range s {
		sum += mapping[val]
		if mapping[prev] < mapping[val] {
			sum -= mapping[prev] * 2
		}
		prev = val
	}
	return sum
}

// Write a function to find the longest common prefix string amongst an array of strings.

// If there is no common prefix, return an empty string "".

func longestCommonPrefix(s []string) string {
	pref := s[0]
	for i := 1; i < len(s); i++ {
		for !strings.HasPrefix(s[i], pref) {
			pref = pref[:len(pref)-1]
			fmt.Println(pref)
		}
	}
	return pref
}

// Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

// The overall run time complexity should be O(log (m+n)).

func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	var newArr []int
	var count float64
	for _, i := range nums1 {
		newArr = append(newArr, i)
	}
	for _, i := range nums2 {
		newArr = append(newArr, i)
	}
	for i := 0; i < len(newArr)-1; i++ {
		for j := 0; j < len(newArr)-i-1; j++ {
			if newArr[j] > newArr[j+1] {
				newArr[j], newArr[j+1] = newArr[j+1], newArr[j]
			}
		}
	}
	if len(newArr)%2 == 0 && len(newArr) > 2 {
		count = (float64(newArr[(len(newArr)/2)-1]) + float64(newArr[(len(newArr)/2)])) / 2
	} else if len(newArr)%2 != 0 && len(newArr) > 2 {
		count = float64(newArr[(len(newArr)/2)+(1/2)])
	} else if len(newArr) == 1 {
		count = float64(newArr[0])
	} else if len(newArr) == 2 {
		count = (float64(newArr[0]) + float64(newArr[1])) / 2
	}
	return float64(count)
}

// The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

// P   A   H   N
// A P L S I I G
// Y   I   R
// And then read line by line: "PAHNAPLSIIGYIR"

// Write the code that will take a string and make this conversion given a number of rows:

// string convert(string s, int numRows);

func convert(s string, numRows int) string {
	if numRows == 1 {
		return s
	}

	resVector := make([]strings.Builder, numRows)
	mod := (numRows - 1) * 2
	for i := 0; i < len(s); i++ {
		index := i % mod
		if index >= numRows {
			index = mod - index
		}
		resVector[index].WriteByte(s[i])
	}

	var res strings.Builder
	for _, r := range resVector {
		res.WriteString(r.String())
	}
	return res.String()
}

// Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.

// Assume the environment does not allow you to store 64-bit integers (signed or unsigned).

func reverse(x int) int {
	var answer int
	for x != 0 {
		answer = answer*10 + x%10
		if answer > 2147483647 || answer < -2147483648 {
			return 0
		}
		fmt.Println(answer, x)
		x /= 10
	}
	return answer
}

// You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

// Find two lines that together with the x-axis form a container, such that the container contains the most water.

// Return the maximum amount of water a container can store.

// Notice that you may not slant the container.

func maxArea(height []int) int {
	start := 0
	end := len(height) - 1
	maxArea := 0
	for start < end {
		currArea := end - start
		if height[start] < height[end] {
			currArea *= height[start]
			start++
		} else {
			currArea *= height[end]
			end--
		}
		if maxArea < currArea {
			maxArea = currArea
		}
	}

	return maxArea
}

// Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

// Notice that the solution set must not contain duplicate triplets.

func threeSum(nums []int) [][]int {
	point := 0
	var droplist [][]int
	n := len(nums)
	sort.Ints(nums)
	for i := range nums {
		if i > 0 && nums[i-1] == nums[i] {
			continue
		}
		j, k := i+1, n-1
		for j < k {
			sum := nums[i] + nums[j] + nums[k]
			if sum > point {
				k--
			} else if sum < point {
				j++
			} else {
				droplist = append(droplist, []int{nums[i], nums[j], nums[k]})
				j++
				for nums[j-1] == nums[j] && j < k {
					j++
				}
			}

		}
	}

	return droplist
}

// You are given an array prices where prices[i] is the price of a given stock on the ith day.

// You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

// Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

func maxProfit(prices []int) int {
	minPrice := prices[0]
	maxProfit := 0
	for i := 1; i < len(prices); i++ {
		price := prices[i]
		if price < minPrice {
			minPrice = price
		} else {
			profit := price - minPrice
			if profit > maxProfit {
				maxProfit = profit
			}
		}
	}
	return maxProfit
}

// Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The order of the elements may be changed. Then return the number of elements in nums which are not equal to val.

// Consider the number of elements in nums which are not equal to val be k, to get accepted, you need to do the following things:

// Change the array nums such that the first k elements of nums contain the elements which are not equal to val. The remaining elements of nums are not important as well as the size of nums.
// Return k.

func removeElement(nums []int, val int) int {
	count := 0
	for _, i := range nums {
		if i != val {
			nums[count] = i
			count++
		}
	}
	return count
}

// Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

// A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

func letterCombinations(digits string) []string {
	if len(digits) == 0 {
		return nil
	}
	class := map[rune]string{
		'2': "abc",
		'3': "def",
		'4': "ghi",
		'5': "jkl",
		'6': "mno",
		'7': "pqrs",
		'8': "tuv",
		'9': "wxyz",
	}
	size := 1
	for _, d := range digits {
		size *= len(class[d])
	}
	res := make([]string, size)
	for _, d := range digits {
		now := class[d]
		size /= len(now)
		for i := range res {
			res[i] += string(now[(i/size)%len(now)])
			fmt.Println(res[i])
		}
	}
	return res

}

// Given the head of a linked list, remove the nth node from the end of the list and return its head.

// Definition for singly-linked list.
type ListNode struct {
	Val  int
	Next *ListNode
}

func removeNthFromEnd(head *ListNode, n int) *ListNode {
	result := &ListNode{Next: head}
	slow, fast := result, result

	for i := 0; i <= n; i++ {
		fast = fast.Next
		fmt.Println("цикл", fast)
	}
	fmt.Println("vne cicle", fast)

	for fast != nil {
		fast = fast.Next
		slow = slow.Next
		fmt.Println(fast)
		fmt.Println(slow)
	}

	slow.Next = slow.Next.Next

	return result.Next

}

//You are given the heads of two sorted linked lists list1 and list2.

// Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.

// Return the head of the merged linked list.
// Definition for singly-linked list.
type ListNode struct {
	Val  int
	Next *ListNode
}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	result := &ListNode{}
	point := result
	for list1 != nil || list2 != nil {
		if list1 == nil {
			point.Next = list2
			list2 = nil
			continue
		}
		if list2 == nil {
			point.Next = list1
			list1 = nil
			continue
		}

		if list1.Val > list2.Val {
			point.Next = list2
			point, list2 = point.Next, list2.Next
		} else {
			point.Next = list1
			point, list1 = point.Next, list1.Next
		}
	}
	return result.Next
}

// Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same. Then return the number of unique elements in nums.

// Consider the number of unique elements of nums to be k, to get accepted, you need to do the following things:

// Change the array nums such that the first k elements of nums contain the unique elements in the order they were present in nums initially. The remaining elements of nums are not important as well as the size of nums.
// Return k.
func removeDuplicates(nums []int) int {
	i := 0
	for j, _ := range nums {
		if nums[i] != nums[j] {
			i++
			nums[i] = nums[j]
		}
	}
	return i + 1
}

// Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

// You must write an algorithm with O(log n) runtime complexity.

func searchInsert(nums []int, target int) int {
	res := 0
	for number, i := range nums {
		if i >= target {
			return number
		} else {
			res = number
		}
	}
	return res + 1
}

// Given a string s consisting of words and spaces, return the length of the last word in the string.

// A word is a maximal
// substring
//  consisting of non-space characters only.

func lengthOfLastWord(s string) int {
	newstr := strings.TrimRight(s, " ")
	arr := strings.Split(newstr, " ")
	return (len(arr[len(arr)-1]))
}

// You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's.

// Increment the large integer by one and return the resulting array of digits.

func plusOne(digits []int) []int {

	for i := len(digits) - 1; i >= 0; i-- {
		if digits[i] < 9 {
			digits[i] = digits[i] + 1
			break
		}
		digits[i] = 0
	}
	if digits[0] == 0 {
		digits = append([]int{1}, digits...)
	}
	return digits
}

// Given a non-negative integer x, return the square root of x rounded down to the nearest integer. The returned integer should be non-negative as well.

// You must not use any built-in exponent function or operator.

// For example, do not use pow(x, 0.5) in c++ or x ** 0.5 in python.

func mySqrt(x int) int {
	first := 0
	second := x

	for first <= second {
		midle := (first + second) / 2
		if midle*midle == x {
			return midle
		} else if midle*midle < x {
			first = midle + 1
		} else {
			second = midle - 1
		}
		fmt.Println(first)
	}
	return second
}

// You are climbing a staircase. It takes n steps to reach the top.

// Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

func climbStairs(n int) int {
	if n < 3 {
		return n
	}

	nowS, nextS := 2, 3
	for i := 3; i <= n; i++ {
		nowS, nextS = nextS, nowS+nextS
	}

	return nowS
}

// Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

// An input string is valid if:

// Open brackets must be closed by the same type of brackets.
// Open brackets must be closed in the correct order.
// Every close bracket has a corresponding open bracket of the same type.

func isValid(s string) bool {
	if s == "" {
		return true
	}

	sLen := len(s)
	s2 := strings.ReplaceAll(s, "()", "")
	fmt.Println("1", s2)
	s2 = strings.ReplaceAll(s2, "[]", "")
	fmt.Println("2", s2)
	s2 = strings.ReplaceAll(s2, "{}", "")
	fmt.Println("3", s2)
	if len(s2) == sLen {
		return false
	}

	return isValid(s2)
}

// Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

func containsDuplicate(nums []int) bool {
	res := ""
	sort.Ints(nums)
	fmt.Println(nums)
	for number, i := range nums {
		val := i
		for j := number; j < len(nums); j++ {
			if j != number {
				if nums[j] == val {
					res = res + "a"
					break
				}
			}

		}
	}
	return len(res) != 0
}

// You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

// Merge nums1 and nums2 into a single array sorted in non-decreasing order.

// The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements
// denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.

func merge(nums1 []int, m int, nums2 []int, n int) {
	var numbers []int
	for i := m; i < len(nums1); i++ {
		numbers = append(numbers, i)
	}
	for j := 0; j <= n-1; j++ {
		nums1[numbers[j]] = nums2[j]
	}
	sort.Ints(nums1)
}

// Given an array nums of size n, return the majority element.

// The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

func majorityElement(nums []int) int {
	class := make(map[int]int)

	for _, i := range nums {
		class[i]++
	}
	fmt.Println(class)

	len := (len(nums) + 1) / 2
	for num, j := range class {
		if j >= len {
			return num
		}
	}

	return nums[0]
}
