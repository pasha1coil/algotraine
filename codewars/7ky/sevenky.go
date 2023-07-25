package seven

import "strings"

// Complete the solution so that it returns true if the first argument(string) passed in ends with the 2nd argument (also a string).

// Examples:

// solution("abc", "bc") // returns true
// solution("abc", "d") // returns false

func solution(str, ending string) bool {

	return strings.HasSuffix(str, ending)

}

// In this kata you should simply determine, whether a given year is a leap year or not. In case you don't know the rules, here they are:

// 	years divisible by 4 are leap years
// 	but years divisible by 100 are not leap years
// 	but years divisible by 400 are leap years
// 	Additional Notes:

// 	Only valid years (positive integers) will be tested, so you don't have to validate them
// 	Examples can be found in the test fixture.

func IsLeapYear(year int) bool {
	if year%4 == 0 && year%100 != 0 || year%400 == 0 {
		return true
	} else {
		return false
	}
}

// Complete the function that accepts a string parameter, and reverses each word in the string. All spaces in the string should be retained.

// Examples
// "This is an example!" ==> "sihT si na !elpmaxe"
// "double  spaces"      ==> "elbuod  secaps"

func ReverseWords(str string) string {
	var rev string
	var word string

	for _, i := range str {
		if i == ' ' {
			rev = rev + word + " "
			word = ""
		} else {
			word = string(i) + word
		}
	}

	return rev + word
}

// Implement a function that accepts 3 integer values a, b, c. The function should return true if a triangle can be built with the sides of given length and false in any other case.

// (In this case, all triangles must have surface greater than 0 to be accepted).

func IsTriangle(a, b, c int) bool {
	return a+b > c && a+c > b && b+c > a

}

// You are going to be given a word. Your job is to return the middle character of the word. If the word's length is odd, return the middle character. If the word's length is even, return the middle 2 characters.

// #Examples:

// Kata.getMiddle("test") should return "es"

// Kata.getMiddle("testing") should return "t"

// Kata.getMiddle("middle") should return "dd"

// Kata.getMiddle("A") should return "A"
// #Input

// A word (string) of length 0 < str < 1000 (In javascript you may get slightly more than 1000 in some test cases due to an error in the test cases). You do not need to test for this. This is only here to tell you that you do not need to worry about your solution timing out.

// #Output

// The middle character(s) of the word represented as a string.

func GetMiddle(s string) string {
	run := []rune(s)
	var newstr string
	if len(s) == 1 {
		newstr = string(run[0])
	} else if len(s) == 2 {
		newstr = string(run[0]) + string(run[1])
	} else {
		if len(s)%2 == 0 {
			newstr = string(run[((len(s)/2)-1)]) + string(run[(((len(s)/2)-1)+1)])
		} else {
			newstr = string(run[(len(s) / 2)])
		}
	}
	return newstr
}

// Your task is to split the chocolate bar of given dimension n x m into small squares. Each square is of size 1x1 and unbreakable. Implement a function that will return minimum number of breaks needed.

// For example if you are given a chocolate bar of size 2 x 1 you can split it to single squares in just one break, but for size 3 x 1 you must do two breaks.

// If input data is invalid you should return 0 (as in no breaks are needed if we do not have any chocolate to split). Input will always be a non-negative integer.

func BreakChocolate(n, m int) int {
	var answer int
	if n > 0 && m > 0 {
		answer = (n * m) - 1
	} else {
		answer = 0
	}
	return answer
}

// Count the number of divisors of a positive integer n.

// Random tests go up to n = 500000.

// Examples (input --> output)
// 4 --> 3 // we have 3 divisors - 1, 2 and 4
// 5 --> 2 // we have 2 divisors - 1 and 5
// 12 --> 6 // we have 6 divisors - 1, 2, 3, 4, 6 and 12
// 30 --> 8 // we have 8 divisors - 1, 2, 3, 5, 6, 10, 15 and 30
// Note you should only return a number, the count of divisors. The numbers between parentheses are shown only for you to see which numbers are counted in each case.

func Divisors(n int) int {
	answer := 1
	for i := 1; i <= n/2; i++ {
		if n%i == 0 {
			answer += 1
		}
	}
	return answer
}

// Deoxyribonucleic acid (DNA) is a chemical found in the nucleus of cells and carries the "instructions" for the development and functioning of living organisms.

// If you want to know more: http://en.wikipedia.org/wiki/DNA

// In DNA strings, symbols "A" and "T" are complements of each other, as "C" and "G". Your function receives one side of the DNA (string, except for Haskell); you need to return the other complementary side. DNA strand is never empty or there is no DNA at all (again, except for Haskell).

// More similar exercise are found here: http://rosalind.info/problems/list-view/ (source)

// Example: (input --> output)

// "ATTGC" --> "TAACG"
// "GTAT" --> "CATA"

func DNAStrand(dna string) string {
	var answer []rune
	t := []rune("T")
	a := []rune("A")
	g := []rune("G")
	c := []rune("C")
	for _, i := range []rune(dna) {
		if i == t[0] {
			answer = append(answer, a[0])
		} else if i == a[0] {
			answer = append(answer, t[0])
		} else if i == g[0] {
			answer = append(answer, c[0])
		} else if i == c[0] {
			answer = append(answer, g[0])
		}
	}
	return string(answer)
}

// Simple, given a string of words, return the length of the shortest word(s).

// String will never be empty and you do not need to account for different data types.

func FindShort(s string) int {
	arr := strings.Split(s, " ")
	count := len(arr[0])
	for _, i := range arr {
		if len(i) < count {
			count = len(i)
		}
	}
	return count
}
