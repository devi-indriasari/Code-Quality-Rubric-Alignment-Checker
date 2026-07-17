public class GradeChecker {
    public static void main(String[] args) {
        int a = 80;
        int b = 75;
        int c = 90;
        int d = (a + b + c) / 3;

        if (d >= 60) {
            System.out.println("Pass");
        } else {
            System.out.println("Fail");
        }
    }
}
