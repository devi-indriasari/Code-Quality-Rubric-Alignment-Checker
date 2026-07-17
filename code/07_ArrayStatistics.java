public class ArrayStatistics {
    public static void main(String[] args) {
        int[] values = {8, 5, 10, 3, 9};
        int sum = 0;
        int max = values[0];
        int min = values[0];
        for (int i = 0; i < values.length; i++) {
            sum += values[i];
            if (values[i] > max) max = values[i];
            if (values[i] < min) min = values[i];
        }
        double average = sum / values.length;
        System.out.println("Average: " + average);
        System.out.println("Max: " + max);
        System.out.println("Min: " + min);
    }
}
