public class ShoppingDiscount {
    public static void main(String[] args) {
        double price = 250000;
        int member = 1;
        double total = price;
        if (member == 1) {
            total = price - price * 0.1;
        }
        if (price > 200000) {
            total = total - 15000;
        }
        System.out.println("Total payment: " + total);
    }
}
