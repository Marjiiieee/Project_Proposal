require('dotenv').config();
const express = require('express');
const mysql = require('mysql');
const bodyParser = require('body-parser');
const session = require('express-session');
const cors = require('cors');
const bcrypt = require('bcrypt');
const multer = require('multer');
const path = require('path');

const app = express();

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(cors());
app.use(express.static(__dirname)); // Serves files from project root
app.use(express.static('uploads')); // Serves uploaded files

// Session Configuration
app.use(session({
    secret: process.env.SESSION_SECRET || 'your_secret_key',
    resave: false,
    saveUninitialized: true,
}));

// MySQL Database Connection Pool
const db = mysql.createPool({
    connectionLimit: 10,
    host: 'localhost',
    user: 'root', // Default XAMPP MySQL user
    password: '', // Default XAMPP MySQL has no password
    database: 'propease'
});

// Check Database Connection
db.getConnection((err, connection) => {
    if (err) {
        console.error('Database connection failed:', err);
    } else {
        console.log('Connected to MySQL database');
        connection.release();
    }
});

// Serve Main Pages
app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'home.html')));
app.get('/signin', (req, res) => res.sendFile(path.join(__dirname, 'signin.html')));
app.get('/account', (req, res) => res.sendFile(path.join(__dirname, 'account.html')));
app.get('/proposals', (req, res) => res.sendFile(path.join(__dirname, 'proposal.html')));

// Login Route
app.post('/login', (req, res) => {
    const { email, password } = req.body;

    if (!email || !password) {
        return res.json({ status: 'error', message: 'Please fill in all fields' });
    }

    const query = 'SELECT password FROM registration WHERE email = ?';
    db.query(query, [email], async (err, results) => {
        if (err) {
            console.error('Database error:', err);
            return res.json({ status: 'error', message: 'Database error occurred' });
        }

        if (results.length > 0) {
            const storedPassword = results[0].password;
            const passwordMatch = await bcrypt.compare(password, storedPassword);

            if (passwordMatch) {
                req.session.email = email;
                return res.json({ status: 'success', redirect: '/account' });
            } else {
                return res.json({ status: 'error', message: 'Invalid password. Please try again.' });
            }
        } else {
            return res.json({ status: 'error', message: 'Email not registered. Please sign up first.' });
        }
    });
});

// Dashboard Route (Redirects after Login)
app.get('/dashboard', (req, res) => {
    if (req.session.email) {
        res.redirect('/account');
    } else {
        res.redirect('/signin');
    }
});

// Logout Route
app.get('/logout', (req, res) => {
    req.session.destroy(() => {
        res.redirect('/signin');
    });
});

// File Upload Handling
const storage = multer.diskStorage({
    destination: './uploads/',
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname));
    },
});

const upload = multer({ storage });

// File Upload API
app.post('/upload', upload.single('file'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }

    const user_email = req.body.user_email;
    const file_name = req.file.originalname;
    const file_path = req.file.filename;

    const sql = "INSERT INTO files (user_email, file_name, file_path, upload_date) VALUES (?, ?, ?, NOW())";
    db.query(sql, [user_email, file_name, file_path], (err, result) => {
        if (err) {
            console.error('Error inserting file:', err);
            return res.status(500).json({ error: 'Database error' });
        }
        res.json({ success: true, filePath: file_path });
    });
});

// Fetch Uploaded Files for User
app.get('/files/:user_email', (req, res) => {
    const sql = "SELECT id, file_name, file_path FROM files WHERE user_email = ?";
    db.query(sql, [req.params.user_email], (err, results) => {
        if (err) {
            console.error('Error fetching files:', err);
            return res.status(500).json({ error: 'Database error' });
        }
        res.json(results);
    });
});


// Start Server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});


function toggleMenu() {
    const menu = document.getElementById('menu-icon');
    menu.classList.toggle('active'); // Toggle class
}
    
document.addEventListener('click', function(event) {
    const menu = document.getElementById('menu-icon');
    const menuIcon = document.querySelector('.menu-icon');
    
if (!menu.contains(event.target) && !menuIcon.contains(event.target)) {
    menu.classList.remove('active'); // Hide menu when clicking outside
    }
});

console.log("Redirecting to file ID:", data.file_id);
