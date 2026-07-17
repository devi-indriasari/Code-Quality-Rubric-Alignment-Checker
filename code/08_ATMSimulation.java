public class ATMSimulation {
    public static void main(String[] args) {
        int balance = 500000;
        int withdraw = 650000;

        if (withdraw <= balance) {
            balance = balance - withdraw;
            System.out.println("Withdraw success");
            System.out.println("Remaining balance: " + balance);
        } else {
            System.out.println("Insufficient balance");
        }
    }
}
