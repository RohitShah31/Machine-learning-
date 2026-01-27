# Lab 2: Case-Based Study of Python Fundamentals

# 1. Store student IDs in a tuple
student_ids = ('S101', 'S102', 'S103', 'S104')

# 2. Store student academic details using a dictionary
students = {
    'S101': {'name': 'Asha', 'assignment': 78, 'test': 80, 'attendance': 92, 'hours': 8},
    'S102': {'name': 'Ravi', 'assignment': 65, 'test': 68, 'attendance': 85, 'hours': 5},
    'S103': {'name': 'Meena', 'assignment': 88, 'test': 90, 'attendance': 96, 'hours': 10},
    'S104': {'name': 'Kiran', 'assignment': 55, 'test': 58, 'attendance': 78, 'hours': 4}
}

# 3. Function to calculate average score
def calculate_average(assignment, test):
    return (assignment + test) / 2

# 4. Function to determine academic risk level
def determine_risk(avg_score, attendance, hours):
    if avg_score < 60 or attendance < 80 or hours < 5:
        return "High Risk"
    elif avg_score < 75:
        return "Moderate Risk"
    else:
        return "Low Risk"

# 5. Process multiple student records
print("\nSTUDENT PERFORMANCE REPORT")
print("-" * 78)

for sid in student_ids:
    student = students[sid]

    avg = calculate_average(student['assignment'], student['test'])
    risk = determine_risk(avg, student['attendance'], student['hours'])

    # 6. Display structured performance report
    print(f"Student ID       : {sid}")
    print(f"Name             : {student['name']}")
    print(f"Assignment Score : {student['assignment']}")
    print(f"Test Score       : {student['test']}")
    print(f"Average Score    : {avg:.2f}")
    print(f"Attendance (%)   : {student['attendance']}")
    print(f"Study Hours/Week : {student['hours']}")
    print(f"Risk Level       : {risk}")
    print("-" * 78)

"""
OUTPUT:
STUDENT PERFORMANCE REPORT
------------------------------------------------------------------------------
Student ID       : S101
Name             : Asha
Assignment Score : 78
Test Score       : 80
Average Score    : 79.00
Attendance (%)   : 92
Study Hours/Week : 8
Risk Level       : Low Risk
------------------------------------------------------------------------------
Student ID       : S102
Name             : Ravi
Assignment Score : 65
Test Score       : 68
Average Score    : 66.50
Attendance (%)   : 85
Study Hours/Week : 5
Risk Level       : Moderate Risk
------------------------------------------------------------------------------
Student ID       : S103
Name             : Meena
Assignment Score : 88
Test Score       : 90
Average Score    : 89.00
Attendance (%)   : 96
Study Hours/Week : 10
Risk Level       : Low Risk
------------------------------------------------------------------------------
Student ID       : S104
Name             : Kiran
Assignment Score : 55
Test Score       : 58
Average Score    : 56.50
Attendance (%)   : 78
Study Hours/Week : 4
Risk Level       : High Risk
------------------------------------------------------------------------------
"""
