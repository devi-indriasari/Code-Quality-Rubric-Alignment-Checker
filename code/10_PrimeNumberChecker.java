public class PrimeNumberChecker {
    public static void main(String[] args) {
        int number = 29;
        boolean prime = true;
        for (int i = 2; i < number; i++) {
            if (number % i == 0) {
                prime = false;
            }
        }
        if (prime) {
            System.out.println(number + " is prime");
        } else {
            System.out.println(number + " is not prime");
        }
    }
}
