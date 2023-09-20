package code

import (
	"container/heap"
	"fmt"
	"math"
	"path"
	"sort"
	"strings"
	"unicode"
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

// Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.
type ListNode struct {
	Val  int
	Next *ListNode
}

func deleteDuplicates(head *ListNode) *ListNode {
	now := head
	for now != nil && now.Next != nil {
		if now.Val == now.Next.Val {
			now.Next = now.Next.Next
		} else {
			now = now.Next
		}
	}
	return head

}

//Given an integer numRows, return the first numRows of Pascal's triangle.

func generate(numRows int) [][]int {
	arr := [][]int{}
	for i := 0; i < numRows; i++ {
		for j := 0; j < i+1; j++ {
			if j == 0 {
				arr = append(arr, []int{1})
			} else {
				if j == i {
					arr[i] = append(arr[i], 1)
				} else {
					arr[i] = append(arr[i], arr[i-1][j]+arr[i-1][j-1])
				}
			}
		}
	}
	return arr
}

// Given the roots of two binary trees p and q, write a function to check if they are the same or not.

// Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if (p == nil) != (q == nil) {
		return false
	}
	if p.Val != q.Val {
		return false
	}
	return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
}

// Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer (similar to C/C++'s atoi function).

// The algorithm for myAtoi(string s) is as follows:

// Read in and ignore any leading whitespace.
// Check if the next character (if not already at the end of the string) is '-' or '+'. Read this character in if it is either. This determines if the final result is negative or positive respectively. Assume the result is positive if neither is present.
// Read in next the characters until the next non-digit character or the end of the input is reached. The rest of the string is ignored.
// Convert these digits into an integer (i.e. "123" -> 123, "0032" -> 32). If no digits were read, then the integer is 0. Change the sign as necessary (from step 2).
// If the integer is out of the 32-bit signed integer range [-231, 231 - 1], then clamp the integer so that it remains in the range. Specifically, integers less than -231 should be clamped to -231, and integers greater than 231 - 1 should be clamped to 231 - 1.
// Return the integer as the final result.
// Note:

// Only the space character ' ' is considered a whitespace character.
// Do not ignore any characters other than the leading whitespace or the rest of the string after the digits.

func myAtoi(s string) int {
	i := 0
	for i < len(s) && s[i] == ' ' {
		i++
	}

	znak := 1
	if i < len(s) && (s[i] == '-' || s[i] == '+') {
		if s[i] == '-' {
			znak = -1
		}
		i++
	}

	num := 0
	for i < len(s) && stepfunction(s[i]) {
		digit := int(s[i] - '0')
		if num > (math.MaxInt32-digit)/10 {
			if znak == 1 {
				return math.MaxInt32
			} else {
				return math.MinInt32
			}
		}
		num = num*10 + digit
		i++
	}

	return num * znak
}

func stepfunction(c byte) bool {
	return c >= '0' && c <= '9'
}

// Given a linked list, swap every two adjacent nodes and return its head.
// You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

// Definition for singly-linked list.
type ListNode struct {
	Val  int
	Next *ListNode
}

func swapPairs(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	head, head.Next, head.Next.Next = head.Next, swapPairs(head.Next.Next), head
	return head
}

// An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5.

// Given an integer n, return true if n is an ugly number.

func isUgly(n int) bool {
	if n <= 0 {
		return false
	}

	for n%2 == 0 {
		n /= 2
	}

	for n%3 == 0 {
		n /= 3
	}

	for n%5 == 0 {
		n /= 5
	}

	return n == 1
}

// A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

// Given a string s, return true if it is a palindrome, or false otherwise.

func isPalindrome(s string) bool {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		fmt.Println(unicode.IsLetter(rune(s[i])))
		for i < j && !(unicode.IsLetter(rune(s[i])) || unicode.IsDigit(rune(s[i]))) {
			i++
		}
		for i < j && !(unicode.IsLetter(rune(s[j])) || unicode.IsDigit(rune(s[j]))) {
			j--
		}
		if i == j {
			break
		}
		if unicode.ToLower(rune(s[i])) != unicode.ToLower(rune(s[j])) {
			return false
		}
	}
	return true
}

// Given a string columnTitle that represents the column title as appears in an Excel sheet, return its corresponding column number.
func titleToNumber(columnTitle string) int {
	result := 0
	for number, i := range columnTitle {
		if len(columnTitle) != 1 {
			result = result + int(math.Pow(26.0, float64(len(columnTitle)-number-1)))*int(i-64)
		} else {
			result = int(i - 64)
		}
	}
	return result
}

// Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.

// You must implement a solution with a linear runtime complexity and use only constant extra space.

func singleNumber(nums []int) int {
	for i := 1; i < len(nums); i++ {
		nums[0] ^= nums[i]
	}
	return nums[0]
}

//Given an integer num, repeatedly add all its digits until the result has only one digit, and return it.

func addDigits(num int) int {
	if num == 0 {
		return 0
	}
	if num%9 == 0 {
		return 9
	}
	return num % 9
}

// There are n employees in a company, numbered from 0 to n - 1. Each employee i has worked for hours[i] hours in the company.

// The company requires each employee to work for at least target hours.

// You are given a 0-indexed array of non-negative integers hours of length n and a non-negative integer target.

// Return the integer denoting the number of employees who worked at least target hours.

func numberOfEmployeesWhoMetTarget(hours []int, target int) int {
	res := 0
	for _, i := range hours {
		if i >= target {
			res = res + 1
		}
	}
	return res
}

// //
func intersection(nums1 []int, nums2 []int) []int {
	class := make(map[int]int)
	for _, i := range nums1 {
		for _, j := range nums2 {
			if i == j {
				class[i]++
			}
		}
	}
	fmt.Println(class)
	var res []int
	for j, _ := range class {
		res = append(res, j)
	}
	return res
}

// You are given an integer array nums consisting of n elements, and an integer k.

// Find a contiguous subarray whose length is equal to k that has the maximum average value and return this value. Any answer with a calculation error less than 10-5 will be accepted.

func findMaxAverage(nums []int, k int) float64 {
	var result, numerator float64 = -797693134862315708145274237, 0
	var l = 0
	divider := float64(k)
	for i := 0; i < len(nums); i++ {
		numerator += float64(nums[i])
		if i < k-1 {
			continue
		}
		for (i-l)+1 > k {
			numerator -= float64(nums[l])
			l++
		}
		result = math.Max(result, numerator/divider)
	}
	return result
}

// A permutation of an array of integers is an arrangement of its members into a sequence or linear order.

// For example, for arr = [1,2,3], the following are all the permutations of arr: [1,2,3], [1,3,2], [2, 1, 3], [2, 3, 1], [3,1,2], [3,2,1].
// The next permutation of an array of integers is the next lexicographically greater permutation of its integer. More formally, if all the permutations of the array are sorted in one container according to their lexicographical order, then the next permutation of that array is the permutation that follows it in the sorted container. If such arrangement is not possible, the array must be rearranged as the lowest possible order (i.e., sorted in ascending order).

// For example, the next permutation of arr = [1,2,3] is [1,3,2].
// Similarly, the next permutation of arr = [2,3,1] is [3,1,2].
// While the next permutation of arr = [3,2,1] is [1,2,3] because [3,2,1] does not have a lexicographical larger rearrangement.
// Given an array of integers nums, find the next permutation of nums.

// The replacement must be in place and use only constant extra memory.

// Example 1:

// Input: nums = [1,2,3]
// Output: [1,3,2]
// Example 2:

// Input: nums = [3,2,1]
// Output: [1,2,3]
// Example 3:

// Input: nums = [1,1,5]
// Output: [1,5,1]

func nextPermutation(nums []int) {
	i := len(nums) - 1
	for ; i > 0; i-- {
		if nums[i-1] < nums[i] {
			for j := len(nums) - 1; j > i-1; j-- {
				if nums[i-1] < nums[j] {
					nums[i-1], nums[j] = nums[j], nums[i-1]
					break
				}
			}
			break
		}
	}
	left, right := i, len(nums)-1
	for ; left < right; left, right = left+1, right-1 {
		nums[left], nums[right] = nums[right], nums[left]
	}
	fmt.Println(nums)
}

// You are given two strings s1 and s2, both of length 4, consisting of lowercase English letters.

// You can apply the following operation on any of the two strings any number of times:

// Choose any two indices i and j such that j - i = 2, then swap the two characters at those indices in the string.
// Return true if you can make the strings s1 and s2 equal, and false otherwise.

func canBeEqual(s1 string, s2 string) bool {
	return s1 == s2 ||
		check(s1, 0, 2) == s2 ||
		check(s1, 1, 3) == s2 ||
		check(check(s1, 0, 2), 1, 3) == s2
}

func check(s1 string, i, j int) string {
	news1 := []byte(s1)
	news1[i], news1[j] = news1[j], news1[i]
	return string(news1)
}

// There is an integer array nums sorted in ascending order (with distinct values).

// Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

// Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

// You must write an algorithm with O(log n) runtime complexity.

func search(nums []int, target int) int {
	for num, value := range nums {
		if value == target {
			return num
		}
	}
	return -1
}

// Given head, the head of a linked list, determine if the linked list has a cycle in it.

// There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

// Return true if there is a cycle in the linked list. Otherwise, return false.

type ListNode struct {
	Val  int
	Next *ListNode
}

func hasCycle(head *ListNode) bool {
	slow_pointer, fast_pointer := head, head
	for fast_pointer != nil && fast_pointer.Next != nil {
		slow_pointer = slow_pointer.Next
		fast_pointer = fast_pointer.Next.Next
		if slow_pointer == fast_pointer {
			return true
		}
	}
	return false
}

// Given two strings s and t, return true if t is an anagram of s, and false otherwise.

// An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

func isAnagram(s string, t string) bool {
	if len(s) != len(t) {
		return false
	}
	dict1 := make(map[string]int)
	dict2 := make(map[string]int)
	for i := 0; i < len(s); i++ {
		dict1[string(s[i])]++
	}
	for i := 0; i < len(s); i++ {
		dict2[string(t[i])]++
	}
	for i := 0; i < len(s); i++ {
		if dict1[string(s[i])] != dict2[string(s[i])] {
			return false
		}
	}
	return true
}

// You are given an m x n integer matrix matrix with the following two properties:

// Each row is sorted in non-decreasing order.
// The first integer of each row is greater than the last integer of the previous row.
// Given an integer target, return true if target is in matrix or false otherwise.

// You must write a solution in O(log(m * n)) time complexity.

func searchMatrix(matrix [][]int, target int) bool {
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			if target == matrix[i][j] {
				return true
			}
		}
	}
	return false
}

// Given an integer array nums and an integer k, return the kth largest element in the array.

// Note that it is the kth largest element in the sorted order, not the kth distinct element.

// Can you solve it without sorting?

// with sorting :(

func findKthLargest(nums []int, k int) int {
	return nums[len(nums)-1]

}

func quicksort(arr []int) []int {
	if len(arr) < 2 {
		return arr
	}
	left := 0
	right := len(arr) - 1
	target := (right + left) / 2
	arr[target], arr[right] = arr[right], arr[target]
	for num, _ := range arr {
		if arr[num] < arr[right] {
			arr[left], arr[num] = arr[num], arr[left]
			left++
		}
	}
	arr[left], arr[right] = arr[right], arr[left]
	quicksort(arr[:left])
	quicksort(arr[left+1:])
	return arr
}

// There is an integer array nums sorted in non-decreasing order (not necessarily with distinct values).

// Before being passed to your function, nums is rotated at an unknown pivot index k (0 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,4,4,5,6,6,7] might be rotated at pivot index 5 and become [4,5,6,6,7,0,1,2,4,4].

// Given the array nums after the rotation and an integer target, return true if target is in nums, or false if it is not in nums.

// You must decrease the overall operation steps as much as possible.

func search(nums []int, target int) bool {
	dict := make(map[int]bool)
	for _, val := range nums {
		dict[val] = true
	}
	return dict[target]
}

// Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.

// If target is not found in the array, return [-1, -1].

// You must write an algorithm with O(log n) runtime complexity.

func searchRange(nums []int, target int) []int {
	var arr []int
	if len(nums) == 0 {
		arr2 := []int{-1, -1}
		return arr2
	} else if len(nums) == 1 && nums[0] == target {
		arr3 := []int{0, 0}
		return arr3
	}
	first := binarysearch(nums, target)
	arr = append(arr, first)
	if first >= 0 {
		arr = append(arr, binarysearch2(nums, target))
	} else {
		arr = append(arr, -1)
	}
	return arr
}

func binarysearch(arr []int, target int) int {
	left := 0
	right := len(arr) - 1
	for left <= right {
		val := left
		if arr[val] < target {
			left = val + 1
		} else if arr[val] > target {
			right = val - 1
		} else {
			return val
		}
	}
	return -1
}

func binarysearch2(arr []int, target int) int {
	left := 0
	right := len(arr) - 1
	for left <= right {
		val := right
		if arr[val] < target {
			left = val + 1
		} else if arr[val] > target {
			right = val - 1
		} else {
			return val
		}
	}
	return -1
}

//УЧИТЬ
// Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

// Each row must contain the digits 1-9 without repetition.
// Each column must contain the digits 1-9 without repetition.
// Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
// Note:

// A Sudoku board (partially filled) could be valid but is not necessarily solvable.
// Only the filled cells need to be validated according to the mentioned rules.

func isValidSudoku(board [][]byte) bool {
	rows := [9][9]bool{}
	column := [9][9]bool{}
	pols := [3][3][9]bool{}
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			kletka := board[i][j]
			if kletka == '.' {
				continue
			}
			chislo := int(kletka-'0') - 1

			if rows[i][chislo] || column[j][chislo] || pols[i/3][j/3][chislo] {
				return false
			}

			rows[i][chislo] = true
			column[j][chislo] = true
			pols[i/3][j/3][chislo] = true
		}
	}
	return true
}

// Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

func permute(nums []int) [][]int {
	if len(nums) == 1 {
		return [][]int{nums}
	}
	if len(nums) == 2 {
		return [][]int{{nums[0], nums[1]}, {nums[1], nums[0]}}
	}
	var perms [][]int
	for i := range nums {
		//i is the elem we skip in subtask
		//permute([2,3]) returns [[2,3],[3,2]]
		for _, p := range permute(append(append([]int{}, nums[:i]...), nums[i+1:]...)) {
			perms = append(perms, append([]int{nums[i]}, p...))
			//[ [1,2,3], [1,3,2] ]
		}
	}
	return perms
}

//Given an m x n matrix, return all elements of the matrix in spiral order.

func spiralOrder(matrix [][]int) []int {
	if matrix == nil && len(matrix) == 0 {
		return []int{}
	}
	var res []int
	top := 0
	bottom := len(matrix) - 1
	left := 0
	right := len(matrix[0]) - 1
	for {
		// вправо
		for i := left; i <= right; i++ {
			res = append(res, matrix[top][i])
		}
		top++
		if top > bottom {
			break
		}

		// вниз
		for i := top; i <= bottom; i++ {
			res = append(res, matrix[i][right])
		}
		right--
		if left > right {
			break
		}

		// влево
		for i := right; i >= left; i-- {
			res = append(res, matrix[bottom][i])
		}
		bottom--
		if top > bottom {
			break
		}

		// вверх
		for i := bottom; i >= top; i-- {
			res = append(res, matrix[i][left])
		}
		left++
		if left > right {
			break
		}
	}
	return res
}

// Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals,
//  and return an array of the non-overlapping intervals that cover all the intervals in the input.

func merge(intervals [][]int) [][]int {
	if len(intervals) <= 1 {
		return intervals
	}

	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	fmt.Println(intervals)

	mergedIntervals := make([][]int, 0, len(intervals))
	mergedIntervals = append(mergedIntervals, intervals[0])

	for _, interval := range intervals[1:] {
		if top := mergedIntervals[len(mergedIntervals)-1]; interval[0] > top[1] {
			mergedIntervals = append(mergedIntervals, interval)
		} else if interval[1] > top[1] {
			top[1] = interval[1]
		}
	}

	return mergedIntervals
}

// You are in a city that consists of n intersections numbered from 0 to n - 1 with bi-directional roads between some intersections. The inputs are generated such that you can reach any intersection from any other intersection and that there is at most one road between any two intersections.

// You are given an integer n and a 2D integer array roads where roads[i] = [ui, vi, timei] means that there is a road between intersections ui and vi that takes timei minutes to travel. You want to know in how many ways you can travel from intersection 0 to intersection n - 1 in the shortest amount of time.

// Return the number of ways you can arrive at your destination in the shortest amount of time. Since the answer may be large, return it modulo 109 + 7.

type Road struct {
	u, v, time int
}

type Node struct {
	index, dist, ways int
}

type PriorityQueue []Node

func (pq PriorityQueue) Len() int { // длина
	return len(pq)
}

func (pq PriorityQueue) Less(i, j int) bool { // проверка на меньше
	return pq[i].dist < pq[j].dist
}

func (pq PriorityQueue) Swap(i, j int) { // замена
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *PriorityQueue) Push(x interface{}) { // запись в список
	node := x.(Node)
	*pq = append(*pq, node)
}

func (pq *PriorityQueue) Pop() interface{} { // удаление по индексу
	old := *pq
	n := len(old)
	node := old[n-1]
	*pq = old[0 : n-1]
	return node
}

func findNumberOfWays(n int, roads [][]int) int {
	adjList := make([][]Road, n)
	for _, road := range roads {
		u, v, time := road[0], road[1], road[2]
		adjList[u] = append(adjList[u], Road{u, v, time})
		adjList[v] = append(adjList[v], Road{v, u, time})
	}

	dist := make([]int, n)
	ways := make([]int, n)
	vis := make([]bool, n)

	dist[0] = 0
	ways[0] = 1

	pq := &PriorityQueue{}
	heap.Push(pq, Node{0, 0, 1})

	for pq.Len() > 0 {
		node := heap.Pop(pq).(Node)
		u := node.index
		fmt.Println(vis)

		if vis[u] {
			continue
		}

		vis[u] = true

		for _, road := range adjList[u] {
			v := road.v
			time := road.time

			if dist[u]+time < dist[v] || dist[v] == 0 {
				dist[v] = dist[u] + time
				ways[v] = ways[u]
				heap.Push(pq, Node{v, dist[v], ways[v]})
			} else if dist[u]+time == dist[v] {
				ways[v] = (ways[v] + ways[u]) % (1e9 + 7)
			}
		}
	}

	return ways[n-1]
}

func countPaths(n int, roads [][]int) int {
	return findNumberOfWays(n, roads)
}

// Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

// [4,5,6,7,0,1,2] if it was rotated 4 times.
// [0,1,2,4,5,6,7] if it was rotated 7 times.
// Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

// Given the sorted rotated array nums of unique elements, return the minimum element of this array.

// You must write an algorithm that runs in O(log n) time.

func findMin(nums []int) int {
	quicksort(nums)
	return nums[0]
}

func quicksort(arr []int) []int {
	if len(arr) < 2 {
		return arr
	}
	left := 0
	right := len(arr) - 1

	target := (left + right) / 2
	arr[target], arr[right] = arr[right], arr[target]
	for i := range arr {
		if arr[i] < arr[right] {
			arr[left], arr[i] = arr[i], arr[left]
			left++
		}
	}
	arr[left], arr[right] = arr[right], arr[left]
	quicksort(arr[:left])
	quicksort(arr[left+1:])
	return arr
}

// Given the root of a complete binary tree, return the number of the nodes in the tree.

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func countNodes(root *TreeNode) int {
	var count int
	if root == nil {
		return count
	}
	count++
	stack := []*TreeNode{root}

	for len(stack) != 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if node.Left != nil {
			count++
			stack = append(stack, node.Left)
		}
		if node.Right != nil {
			count++
			stack = append(stack, node.Right)
		}
	}
	return count
}

// Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

// There is only one repeated number in nums, return this repeated number.

// You must solve the problem without modifying the array nums and uses only constant extra space.

func findDuplicate(nums []int) int {
	dict := make(map[int]int)
	for _, i := range nums {
		dict[i]++
	}
	for _, i := range nums {
		if dict[i] > 1 {
			return i
		}
	}
	return 1
}

// Given two integer arrays nums1 and nums2, return an array of their intersection. Each element in the result must appear
// as many times as it shows in both arrays and you may return the result in any order.

func intersect(nums1 []int, nums2 []int) []int {
	m := make(map[int]int)
	array := make([]int, 0)
	for i := 0; i < len(nums1); i++ {
		m[nums1[i]]++
	}
	for j := 0; j < len(nums2); j++ {
		fmt.Println(m[nums2[j]])
		if val, ok := m[nums2[j]]; ok {
			if val >= 1 {
				m[nums2[j]] -= 1
				array = append(array, nums2[j])
			}
		}
	}
	return array
}

// Учитывая два целочисленных массива preorder, inorderгде preorder—
// предварительный обход двоичного дерева, а inorder— неупорядоченный обход того же дерева, постройте и верните двоичное дерево .

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}
	i := 0
	for preorder[0] != inorder[i] {
		i++
	}
	return &TreeNode{preorder[0], buildTree(preorder[1:i+1], inorder[:i]), buildTree(preorder[i+1:], inorder[i+1:])}
}

func main() {
	fmt.Println(buildTree([]int{3, 9, 20, 15, 7}, []int{9, 3, 15, 20, 7}))
}

// Учитывая rootдвоичное дерево и целое число targetSum, возвратите true, если дерево имеет путь от корня к листу,
// так что сложение всех значений вдоль пути равно targetSum.
// Лист — это узел, не имеющий дочерних элементов .

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func hasPathSum(root *TreeNode, targetSum int) bool {
	var result = false
	Helper(root, targetSum, &result)

	return result

}

func Helper(root *TreeNode, targetSum int, result *bool) {
	if *result == true {
		*result = true
	}
	if root == nil {
		return
	}
	if root.Left != nil {
		root.Left.Val = root.Left.Val + root.Val
		Helper(root.Left, targetSum, result)
	}
	if root.Right != nil {
		root.Right.Val = root.Right.Val + root.Val
		Helper(root.Right, targetSum, result)
	}
	if root.Right == nil && root.Left == nil {
		if root.Val == targetSum {
			*result = true
		}

	}
	return
}

// Given the root of a binary tree, return the preorder traversal of its nodes' values.

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type Sl struct {
	array []int
}

func preorderTraversal(root *TreeNode) []int {
	arr := Sl{}
	arr.helper(root)

	return arr.array
}

func (arr *Sl) helper(root *TreeNode) {
	if root != nil {
		arr.array = append(arr.array, root.Val)
		arr.helper(root.Left)
		arr.helper(root.Right)
	}
}

// Given the root of a binary tree, return the postorder traversal of its nodes' values.

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type Sl struct {
	array []int
}

func postorderTraversal(root *TreeNode) []int {
	arr := Sl{}
	arr.helper(root)
	return arr.array
}

func (arr *Sl) helper(root *TreeNode) {
	if root != nil {
		arr.helper(root.Left)                   // Рекурсивно обходим левое поддерево
		arr.helper(root.Right)                  // Рекурсивно обходим правое поддерево
		arr.array = append(arr.array, root.Val) // Посещаем корневой узел
	}
}

// Given a string path, which is an absolute path (starting with a slash '/') to a file or directory in a Unix-style file system, convert it to the simplified canonical path.

// In a Unix-style file system, a period '.' refers to the current directory, a double period '..' refers to the directory up a level, and any multiple consecutive slashes (i.e. '//') are treated as a single slash '/'. For this problem, any other format of periods such as '...' are treated as file/directory names.

// The canonical path should have the following format:

// The path starts with a single slash '/'.
// Any two directories are separated by a single slash '/'.
// The path does not end with a trailing '/'.
// The path only contains the directories on the path from the root directory to the target file or directory (i.e., no period '.' or double period '..')
// Return the simplified canonical path.

func simplifyPath(p string) string {
	return path.Clean(p)
}

// Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.

func setZeroes(matrix [][]int) {
	var index []int
	var str []int
	razmer := len(matrix)
	razmerstr := len(matrix[0])
	for i := 0; i < razmer; i++ {
		for j := 0; j < razmerstr; j++ {
			if matrix[i][j] == 0 {
				index = append(index, j)
				str = append(str, i)
			}
		}
	}
	for i := 0; i < razmer; i++ {
		for j := 0; j < len(index); j++ {
			matrix[i][index[j]] = 0
		}
	}
	for i := 0; i < len(str); i++ {
		for j := 0; j < len(matrix[str[i]]); j++ {
			matrix[str[i]][j] = 0
		}
	}
	fmt.Println(matrix)
}

// Given two integers n and k, return all possible combinations of k numbers chosen from the range [1, n].

// You may return the answer in any order.

func combine(n int, k int) [][]int {
	res, arr, i := [][]int{}, make([]int, k), 0

	for i >= 0 {
		arr[i]++
		if arr[i] > n {
			i--
		} else if i == k-1 {
			res = append(res, append([]int{}, arr...))
		} else {
			i++
			arr[i] = arr[i-1]
		}
	}

	return res
}

// Given the root of a binary tree, return the inorder traversal of its nodes' values.
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func inorderTraversal(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	var arr []int
	helper(root, &arr)
	return arr
}

func helper(root *TreeNode, arr *[]int) {
	if root == nil {
		return
	}
	helper(root.Left, arr)
	*arr = append(*arr, root.Val)
	helper(root.Right, arr)
}

// Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

var answer bool

func isSymmetric(root *TreeNode) bool {
	answer = true
	if root != nil {
		recurseSymmetric(root.Left, root.Right)
	}
	return answer
}

func recurseSymmetric(root1, root2 *TreeNode) {
	if root1 == nil && root2 == nil || answer == false {
		return
	}
	if root1 == nil || root2 == nil || root1.Val != root2.Val {
		answer = false
		return
	}
	recurseSymmetric(root1.Left, root2.Right)
	recurseSymmetric(root1.Right, root2.Left)
}

// Given an integer rowIndex, return the rowIndexth (0-indexed) row of the Pascal's triangle.

// In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:

func getRow(rowIndex int) []int {
	row := make([]int, rowIndex+1)
	row[0] = 1
	for i := 1; i <= rowIndex; i++ {
		tmp := uint64(row[i-1])
		tmp = tmp * uint64(rowIndex+1-i)
		tmp = tmp / uint64(i)
		fmt.Println(row)
		row[i] = int(tmp)
	}
	return row
}

// Given a binary tree, find its minimum depth.

// The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

func minDepth(root *TreeNode) int {
	k := math.MaxInt64
	find(root, &k, 0)
	if k == math.MaxInt64 {
		return 0
	}
	return k
}

func find(root *TreeNode, re *int, cur int) {
	if root == nil {
		return
	}
	if root.Left != nil {
		find(root.Left, re, cur+1)
		// return
	}
	if root.Right != nil {
		find(root.Right, re, cur+1)
		// return
	}
	if root.Right == nil && root.Left == nil {
		if cur+1 < *re {
			*re = cur + 1
		}
		return
	}
}

// Write an algorithm to determine if a number n is happy.

// A happy number is a number defined by the following process:

// Starting with any positive integer, replace the number by the sum of the squares of its digits.
// Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
// Those numbers for which this process ends in 1 are happy.
// Return true if n is a happy number, and false if not.

func isHappy(n int) bool {
	switch n {
	case 1:
		return true
	case 0, 2, 3, 4, 5, 6, 8, 9:
		return false
	default:
		return isHappy(Helper(n))
	}
}

func Helper(n int) int {
	var out int
	var last int
	for n > 0 {
		last = n % 10
		out += last * last
		n /= 10
	}
	return out
}
