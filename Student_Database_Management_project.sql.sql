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

-- Insert new students
INSERT INTO Students (FirstName, LastName, DateOfBirth, Gender, GPA)
VALUES 
    ('John', 'Doe', '1995-05-15', 'Male', 3.75),
    ('Jane', 'Smith', '1997-08-21', 'Female', 3.90),
    ('Mike', 'Donovan', '1996-06-25', 'Other', 3.20);

-- Update a student's GPA
UPDATE Students
SET GPA = 3.80
WHERE StudentID = 1;

-- Select all students
SELECT * FROM Students;

-- Select students with a GPA greater than 3.5
SELECT * FROM Students WHERE GPA > 3.5;

-- Select a specific student
SELECT * FROM Students WHERE StudentID = 1;

-- Delete a student
DELETE FROM Students WHERE StudentID = 3;

-- Creating Indexes to improve query performance
CREATE INDEX idx_last_name ON Students (LastName);

-- Drop the existing database for courses if it exists to start fresh
DROP DATABASE IF EXISTS StudentDataB;

-- Create a separate database for courses if needed
CREATE DATABASE StudentDataB;
USE StudentDataB;

-- Create a table to store student information
CREATE TABLE IF NOT EXISTS Students (
    StudentID INT PRIMARY KEY AUTO_INCREMENT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    DateOfBirth DATE,
    Email VARCHAR(100)
);

-- Create a table to store course information
CREATE TABLE IF NOT EXISTS Courses (
    CourseID INT PRIMARY KEY AUTO_INCREMENT,
    CourseName VARCHAR(100)
);

-- Create a junction table to represent student-course relationships
CREATE TABLE IF NOT EXISTS StudentCourses (
    StudentCourseID INT PRIMARY KEY AUTO_INCREMENT,
    StudentID INT,
    CourseID INT,
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (CourseID) REFERENCES Courses(CourseID)
);

-- Insert sample data into Students table
INSERT INTO Students (FirstName, LastName, DateOfBirth, Email)
VALUES
    ('John', 'Doe', '1995-03-15', 'john.doe@example.com'),
    ('Jane', 'Smith', '1997-07-20', 'jane.smith@example.com'),
    ('Bob', 'Johnson', '1998-01-10', 'bob.johnson@example.com');

-- Insert sample data into Courses table
INSERT INTO Courses (CourseName)
VALUES
    ('Math 101'),
    ('History 101'),
    ('Science 101');

-- Assign students to courses in the junction table
INSERT INTO StudentCourses (StudentID, CourseID)
VALUES
    (1, 1), -- John in Math 101
    (1, 2), -- John in History 101
    (2, 2), -- Jane in History 101
    (3, 3); -- Bob in Science 101

-- Display all students
SELECT * FROM Students;

-- Display all courses
SELECT * FROM Courses;

-- Display all student-course relationships
SELECT * FROM StudentCourses;
