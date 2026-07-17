public class PasswordStrength {
    public static void main(String[] args) {
        String password = "abc123";
        int score = 0;

        if (password.length() >= 8) score++;
        if (password.matches(".*[A-Z].*")) score++;
        if (password.matches(".*[0-9].*")) score++;
        if (password.matches(".*[!@#$%^&*].*")) score++;

        if (score >= 3) {
            System.out.println("Strong password");
        } else {
            System.out.println("Weak password");
        }
    }
}
