-- Drop the existing database if it exists to start fresh
DROP DATABASE IF EXISTS StudentDB;

-- Create the main database for student management
CREATE DATABASE StudentDB;
USE StudentDB;

-- Create a table to store student information
CREATE TABLE IF NOT EXISTS Students (
    StudentID INT AUTO_INCREMENT PRIMARY KEY,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    DateOfBirth DATE,
    Gender ENUM('Male', 'Female', 'Other'),
    GPA DECIMAL(3, 2)
);

-- Create a table to store course information
CREATE TABLE IF NOT EXISTS Courses (
    CourseID INT AUTO_INCREMENT PRIMARY KEY,
    CourseName VARCHAR(100),
    Credits INT
);

-- Create a junction table to represent student-course relationships
CREATE TABLE IF NOT EXISTS StudentCourses (
    StudentCourseID INT AUTO_INCREMENT PRIMARY KEY,
    StudentID INT,
    CourseID INT,
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (CourseID) REFERENCES Courses(CourseID)
);

-- Insert sample data into Students table (100+ student records)
INSERT INTO Students (FirstName, LastName, DateOfBirth, Gender, GPA) VALUES 
    ('John', 'Doe', '1995-05-15', 'Male', 3.75),
    ('Jane', 'Smith', '1997-08-21', 'Female', 3.90),
    ('Mike', 'Donovan', '1996-06-25', 'Other', 3.20),
    ('Alice', 'Johnson', '1994-03-10', 'Female', 3.60),
    ('Bob', 'Williams', '1993-07-30', 'Male', 2.80),
    ('Charlie', 'Brown', '1995-11-12', 'Male', 3.50),
    ('Daisy', 'Martinez', '1996-01-15', 'Female', 3.85),
    ('Ethan', 'Davis', '1994-12-01', 'Male', 2.90),
    ('Fiona', 'Garcia', '1995-10-22', 'Female', 3.40),
    ('George', 'Hernandez', '1997-02-17', 'Male', 3.95),
    ('Hannah', 'Lopez', '1993-09-18', 'Female', 3.60),
    ('Ian', 'Gonzalez', '1996-04-20', 'Male', 2.70),
    ('Jasmine', 'Wilson', '1995-06-11', 'Female', 3.80),
    ('Kevin', 'Anderson', '1994-08-14', 'Male', 3.10),
    ('Laura', 'Thomas', '1996-03-05', 'Female', 3.65),
    ('Michael', 'Taylor', '1995-12-30', 'Male', 2.85),
    ('Nora', 'Moore', '1994-07-24', 'Female', 3.75),
    ('Oliver', 'Jackson', '1995-01-01', 'Male', 3.55),
    ('Paula', 'White', '1996-05-06', 'Female', 3.25),
    ('Quinn', 'Harris', '1994-10-09', 'Other', 3.95),
    ('Ryan', 'Martin', '1995-07-08', 'Male', 2.60),
    ('Sophia', 'Thompson', '1996-08-12', 'Female', 3.80),
    ('Tyler', 'Garcia', '1994-04-21', 'Male', 3.15),
    ('Ursula', 'Martinez', '1995-09-19', 'Female', 3.45),
    ('Victor', 'Robinson', '1996-02-22', 'Male', 3.70),
    ('Wendy', 'Clark', '1995-11-11', 'Female', 3.25),
    ('Xavier', 'Rodriguez', '1994-12-30', 'Male', 3.85),
    ('Yasmine', 'Lewis', '1996-03-03', 'Female', 2.95),
    ('Zach', 'Walker', '1993-10-15', 'Male', 3.60),
    ('Ashley', 'Hall', '1995-05-23', 'Female', 3.80),
    ('Brad', 'Allen', '1996-06-30', 'Male', 3.40),
    ('Claire', 'Young', '1994-08-19', 'Female', 3.65),
    ('Daniel', 'Hernandez', '1995-04-12', 'Male', 3.20),
    ('Eva', 'King', '1997-12-05', 'Female', 3.90),
    ('Finn', 'Scott', '1995-01-17', 'Male', 3.00),
    ('Gina', 'Green', '1996-10-27', 'Female', 3.75),
    ('Henry', 'Adams', '1995-02-14', 'Male', 2.80),
    ('Isabella', 'Baker', '1994-09-22', 'Female', 3.55),
    ('Jack', 'Gonzalez', '1997-03-03', 'Male', 3.30),
    ('Kathy', 'Nelson', '1995-12-20', 'Female', 3.85),
    ('Leo', 'Carter', '1996-07-06', 'Male', 3.40),
    ('Mia', 'Mitchell', '1994-11-18', 'Female', 3.65),
    ('Noah', 'Perez', '1995-08-04', 'Male', 2.95),
    ('Olivia', 'Roberts', '1996-02-28', 'Female', 3.90),
    ('Paul', 'Turner', '1994-01-09', 'Male', 3.10),
    ('Queen', 'Phillips', '1995-04-25', 'Other', 3.50),
    ('Ray', 'Campbell', '1996-06-14', 'Male', 3.80),
    ('Samantha', 'Parker', '1994-03-07', 'Female', 3.20),
    ('Travis', 'Evans', '1995-05-31', 'Male', 3.45),
    ('Uma', 'Edwards', '1996-01-04', 'Female', 3.70),
    ('Vince', 'Collins', '1994-12-17', 'Male', 2.85),
    ('Willow', 'Stewart', '1995-10-29', 'Female', 3.75),
    ('Xena', 'Sanchez', '1996-08-10', 'Other', 3.25),
    ('Yusuf', 'Morris', '1994-02-18', 'Male', 3.60),
    ('Zoe', 'Rogers', '1995-11-15', 'Female', 3.80),
    ('Amelia', 'Hall', '1995-01-01', 'Female', 3.65),
    ('Ben', 'Rodriguez', '1994-05-15', 'Male', 2.95),
    ('Cathy', 'Lee', '1996-04-12', 'Female', 3.50),
    ('David', 'Kim', '1995-02-25', 'Male', 3.90),
    ('Ella', 'Park', '1996-03-10', 'Female', 3.80),
    ('Felix', 'Wong', '1994-11-20', 'Male', 3.30),
    ('Grace', 'Nguyen', '1995-12-18', 'Female', 3.75),
    ('Henry', 'Lo', '1996-05-15', 'Male', 3.20),
    ('Ivy', 'Ng', '1995-07-29', 'Female', 3.95),
    ('Jack', 'Tan', '1994-03-08', 'Male', 2.85),
    ('Luna', 'Koh', '1996-08-05', 'Female', 3.65),
    ('Mark', 'Sim', '1995-01-11', 'Male', 3.40),
    ('Nina', 'Cheng', '1996-06-01', 'Female', 3.70),
    ('Oscar', 'Tay', '1994-12-12', 'Male', 3.10),
    ('Penny', 'Goh', '1995-03-03', 'Female', 3.60),
    ('Quincy', 'Lim', '1996-02-21', 'Other', 2.95),
    ('Rita', 'Phua', '1995-08-08', 'Female', 3.75),
    ('Sam', 'Foo', '1994-07-25', 'Male', 3.40),
    ('Tina', 'Ong', '1995-11-14', 'Female', 3.85),
    ('Victor', 'Yeo', '1996-09-17', 'Male', 3.60),
    ('Winnie', 'Lim', '1994-05-09', 'Female', 3.90),
    ('Xander', 'Tan', '1995-01-28', 'Male', 3.25),
    ('Yara', 'Chua', '1996-04-30', 'Female', 3.45),
    ('Zara', 'Kumar', '1995-06-12', 'Female', 3.70),
    ('Alan', 'Rai', '1994-12-07', 'Male', 2.80),
    ('Bella', 'Nair', '1995-03-20', 'Female', 3.55),
    ('Chloe', 'Yadav', '1996-05-16', 'Female', 3.85),
    ('Daniel', 'Bhatia', '1994-11-11', 'Male', 3.20),
    ('Ethan', 'Patel', '1995-08-03', 'Male', 3.90),
    ('Fiona', 'Singh', '1996-09-19', 'Female', 3.65),
    ('George', 'Khan', '1994-01-29', 'Male', 3.10),
    ('Helen', 'Sethi', '1995-02-08', 'Female', 3.75),
    ('Ian', 'Sharma', '1996-06-14', 'Male', 3.80),
    ('Judy', 'Bansal', '1994-03-04', 'Female', 3.40),
    ('Kevin', 'Menon', '1995-10-17', 'Male', 3.60),
    ('Lisa', 'Mishra', '1996-12-25', 'Female', 3.80),
    ('Mike', 'Roy', '1994-08-05', 'Male', 3.25),
    ('Nina', 'Gupta', '1995-04-02', 'Female', 3.45),
    ('Omar', 'Jain', '1996-01-18', 'Male', 3.50),
    ('Pooja', 'Kapoor', '1995-11-11', 'Female', 3.65),
    ('Raj', 'Kumar', '1994-05-15', 'Male', 3.20),
    ('Sita', 'Sharma', '1995-07-28', 'Female', 3.75),
    ('Tariq', 'Choudhury', '1996-02-04', 'Male', 3.85),
    ('Uma', 'Sethi', '1995-09-11', 'Female', 3.30),
    ('Vikram', 'Mehta', '1994-10-20', 'Male', 3.15),
    ('Wajid', 'Qureshi', '1995-08-15', 'Male', 3.45),
    ('Zain', 'Chowdhury', '1996-03-03', 'Male', 3.55);

-- Insert sample data into Courses table (10+ course details)
INSERT INTO Courses (CourseName, Credits) VALUES 
    ('Database Systems', 3),
    ('Data Structures', 4),
    ('Machine Learning', 3),
    ('Artificial Intelligence', 4),
    ('Web Development', 3),
    ('Computer Networks', 3),
    ('Operating Systems', 4),
    ('Software Engineering', 3),
    ('Human-Computer Interaction', 3),
    ('Digital Signal Processing', 4),
    ('Cybersecurity', 3);

-- Enroll students in courses (more than 100 student-course enrollments)
INSERT INTO StudentCourses (StudentID, CourseID) VALUES
    (1, 1), (1, 2), (2, 3), (2, 4), (3, 1), 
    (4, 2), (4, 3), (5, 1), (6, 5), (7, 6),
    (8, 7), (9, 8), (10, 9), (11, 10), (12, 1),
    (13, 2), (14, 3), (15, 4), (16, 5), (17, 6),
    (18, 7), (19, 8), (20, 9), (21, 10), (22, 1),
    (23, 2), (24, 3), (25, 4), (26, 5), (27, 6),
    (28, 7), (29, 8), (30, 9), (31, 10), (32, 1),
    (33, 2), (34, 3), (35, 4), (36, 5), (37, 6),
    (38, 7), (39, 8), (40, 9), (41, 10), (42, 1),
    (43, 2), (44, 3), (45, 4), (46, 5), (47, 6),
    (48, 7), (49, 8), (50, 9), (51, 10), (52, 1),
    (53, 2), (54, 3), (55, 4), (56, 5), (57, 6),
    (58, 7), (59, 8), (60, 9), (61, 10), (62, 1),
    (63, 2), (64, 3), (65, 4), (66, 5), (67, 6),
    (68, 7), (69, 8), (70, 9), (71, 10), (72, 1),
    (73, 2), (74, 3), (75, 4), (76, 5), (77, 6),
    (78, 7), (79, 8), (80, 9), (81, 10), (82, 1),
    (83, 2), (84, 3), (85, 4), (86, 5), (87, 6),
    (88, 7), (89, 8), (90, 9), (91, 10), (92, 1),
    (93, 2), (94, 3), (95, 4), (96, 5), (97, 6),
    (98, 7), (99, 8), (100, 9), (1, 10), (2, 1),
    (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7),
    (9, 8), (10, 9), (11, 10), (12, 1), (13, 2),
    (14, 3), (15, 4), (16, 5), (17, 6), (18, 7),
    (19, 8), (20, 9), (21, 10), (22, 1), (23, 2),
    (24, 3), (25, 4), (26, 5), (27, 6), (28, 7),
    (29, 8), (30, 9), (31, 10), (32, 1), (33, 2),
    (34, 3), (35, 4), (36, 5), (37, 6), (38, 7),
    (39, 8), (40, 9), (41, 10), (42, 1), (43, 2),
    (44, 3), (45, 4), (46, 5), (47, 6), (48, 7),
    (49, 8), (50, 9), (51, 10), (52, 1), (53, 2),
    (54, 3), (55, 4), (56, 5), (57, 6), (58, 7),
    (59, 8), (60, 9), (61, 10), (62, 1), (63, 2),
    (64, 3), (65, 4), (66, 5), (67, 6), (68, 7),
    (69, 8), (70, 9), (71, 10), (72, 1), (73, 2),
    (74, 3), (75, 4), (76, 5), (77, 6), (78, 7),
    (79, 8), (80, 9), (81, 10), (82, 1), (83, 2),
    (84, 3), (85, 4), (86, 5), (87, 6), (88, 7),
    (89, 8), (90, 9), (91, 10), (92, 1), (93, 2),
    (94, 3), (95, 4), (96, 5), (97, 6), (98, 7),
    (99, 8), (100, 9), (1, 10);

-- Sample SQL Queries
-- Query 1: Get all students
SELECT * FROM Students;

-- Query 2: Get all courses
SELECT * FROM Courses;

-- Query 3: Get students enrolled in a specific course
SELECT s.FirstName, s.LastName 
FROM Students s 
JOIN StudentCourses sc ON s.StudentID = sc.StudentID 
JOIN Courses c ON sc.CourseID = c.CourseID 
WHERE c.CourseName = 'Machine Learning';

-- Query 4: Get courses for a specific student
SELECT c.CourseName 
FROM Courses c 
JOIN StudentCourses sc ON c.CourseID = sc.CourseID 
JOIN Students s ON sc.StudentID = s.StudentID 
WHERE s.FirstName = 'John' AND s.LastName = 'Doe';

-- Query 5: Get average GPA of students
SELECT AVG(GPA) AS AverageGPA FROM Students;

-- Query 6: Get students with GPA above a certain threshold
SELECT * FROM Students WHERE GPA > 3.5;

-- Query 7: Count number of students per course
SELECT c.CourseName, COUNT(sc.StudentID) AS StudentCount 
FROM Courses c 
LEFT JOIN StudentCourses sc ON c.CourseID = sc.CourseID 
GROUP BY c.CourseName;

-- Query 8: Get students not enrolled in any course
SELECT s.FirstName, s.LastName 
FROM Students s 
LEFT JOIN StudentCourses sc ON s.StudentID = sc.StudentID 
WHERE sc.CourseID IS NULL;

-- Query 9: Get the highest GPA
SELECT MAX(GPA) AS HighestGPA FROM Students;

-- Query 10: Get the lowest GPA
SELECT MIN(GPA) AS LowestGPA FROM Students;

-- Query 11: Get students sorted by GPA
SELECT * FROM Students ORDER BY GPA DESC;


-- Query 13: Get course details for each student
SELECT s.FirstName, s.LastName, c.CourseName 
FROM Students s 
JOIN StudentCourses sc ON s.StudentID = sc.StudentID 
JOIN Courses c ON sc.CourseID = c.CourseID;

-- Query 14: Get students who are enrolled in more than one course
SELECT s.FirstName, s.LastName 
FROM Students s 
JOIN StudentCourses sc ON s.StudentID = sc.StudentID 
GROUP BY s.StudentID 
HAVING COUNT(sc.CourseID) > 1;

-- Query 15: Get students by gender
SELECT * FROM Students WHERE Gender = 'Male';

-- Query 16: Get courses with more than 10 students
SELECT c.CourseName, COUNT(sc.StudentID) AS StudentCount 
FROM Courses c 
JOIN StudentCourses sc ON c.CourseID = sc.CourseID 
GROUP BY c.CourseName 
HAVING COUNT(sc.StudentID) > 10;

-- Query 17: Get students by first name
SELECT * FROM Students WHERE FirstName LIKE 'J%';

-- Query 18: Get course count
SELECT COUNT(*) AS CourseCount FROM Courses;

-- Query 19: Get all students and their courses
SELECT s.FirstName, s.LastName, GROUP_CONCAT(c.CourseName) AS Courses 
FROM Students s 
LEFT JOIN StudentCourses sc ON s.StudentID = sc.StudentID 
LEFT JOIN Courses c ON sc.CourseID = c.CourseID 
GROUP BY s.StudentID;

-- Query 20: Get total student count
SELECT COUNT(*) AS TotalStudents FROM Students;
