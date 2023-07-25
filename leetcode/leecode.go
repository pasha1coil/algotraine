package code

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
