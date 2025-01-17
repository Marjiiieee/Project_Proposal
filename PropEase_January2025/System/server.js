document.getElementById('login-form').addEventListener('submit', function(event) {
    event.preventDefault();

    // Get the input values
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    // Simple validation
    if (email === '' || password === '') {
        alert('Please enter both email and password');
        return;
    }

    // Store user data in localStorage (for demo purposes)
    localStorage.setItem('userEmail', email);
    localStorage.setItem('userPassword', password);

    // Redirect to account page
    window.location.href = 'account.html';
});

// Display user data on account page
window.addEventListener('DOMContentLoaded', (event) => {
    if (window.location.pathname.endsWith('account.html')) {
        const email = localStorage.getItem('userEmail');
        const password = localStorage.getItem('userPassword');

        if (email && password) {
            document.getElementById('name').textContent = `Name: ${email.split('@')[0]}`;
            document.getElementById('email').textContent = `Email: ${email}`;
            document.getElementById('password').textContent = `Password: ${'*'.repeat(password.length)}`;
        } else {
            window.location.href = 'main.html'; // Redirect to login if no data found
        }
    }
});
