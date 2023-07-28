package code

import (
	"fmt"
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


