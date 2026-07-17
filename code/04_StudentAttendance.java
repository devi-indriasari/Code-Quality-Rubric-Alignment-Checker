public class StudentAttendance {
    public static void main(String[] args) {
        int totalMeetings = 16;
        int attend = 11;
        double percentage = attend / totalMeetings * 100;

        if (percentage >= 75) {
            System.out.println("Allowed to take exam");
        } else {
            System.out.println("Not allowed to take exam");
        }
    }
}
