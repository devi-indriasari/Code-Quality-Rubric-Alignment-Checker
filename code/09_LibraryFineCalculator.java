public class LibraryFineCalculator {
    public static void main(String[] args) {
        int lateDays = 9;
        int fine = 0;
        if (lateDays <= 0) {
            fine = 0;
        } else if (lateDays <= 7) {
            fine = lateDays * 1000;
        } else {
            fine = 7000 + (lateDays - 7) * 2000;
        }
        System.out.println("Fine: " + fine);
    }
}
