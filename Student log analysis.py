import re
from collections import defaultdict
from io import StringIO

# --------------------------------
# Student Class
# --------------------------------
class Student:
    def __init__(self, student_id, name):
        self.student_id = student_id
        self.name = name
        self.activities = []

    def add_activity(self, activity, date, time):
        self.activities.append((activity, date, time))

    def activity_summary(self):
        summary = defaultdict(int)
        for activity, _, _ in self.activities:
            summary[activity] += 1
        return summary


# --------------------------------
# Embedded Input Log Data
# --------------------------------
log_data = """\
S101 | Asha | LOGIN | 2025-03-10 | 09:15
S102 | Ravi | SUBMIT_ASSIGNMENT | 2025-03-10 | 22:40
S101 | Asha | LOGOUT | 2025-03-10 | 11:30
S101 | Asha | LOGIN | 2025-03-11 | 10:00
"""

# --------------------------------
# Regex Patterns
# --------------------------------
LOG_PATTERN = r"^(S\d+)\s*\|\s*(\w+)\s*\|\s*(LOGIN|LOGOUT|SUBMIT_ASSIGNMENT)\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d{2}:\d{2})$"


# --------------------------------
# Generator Function
# --------------------------------
def log_reader(data):
    file = StringIO(data)
    for line in file:
        line = line.strip()
        try:
            match = re.match(LOG_PATTERN, line)
            if not match:
                raise ValueError("Invalid log entry")

            yield match.groups()

        except Exception:
            print("Invalid entry skipped:", line)


# --------------------------------
# Main Logic
# --------------------------------
students = {}
daily_activity = defaultdict(int)
login_tracker = defaultdict(int)

for student_id, name, activity, date, time in log_reader(log_data):
    if student_id not in students:
        students[student_id] = Student(student_id, name)

    students[student_id].add_activity(activity, date, time)
    daily_activity[date] += 1

    if activity == "LOGIN":
        login_tracker[student_id] += 1
    elif activity == "LOGOUT":
        login_tracker[student_id] -= 1


# --------------------------------
# Display Report
# --------------------------------
print("\n----- STUDENT ACTIVITY REPORT -----")

for student in students.values():
    summary = student.activity_summary()

    print("\nStudent ID:", student.student_id)
    print("Name:", student.name)
    print("Total Logins:", summary.get("LOGIN", 0))
    print("Total Submissions:", summary.get("SUBMIT_ASSIGNMENT", 0))

print("\n----- DAILY ACTIVITY STATISTICS -----")
for date, count in daily_activity.items():
    print(date, ":", count, "activities")

print("\n----- ABNORMAL BEHAVIOR -----")
for student_id, count in login_tracker.items():
    if count > 0:
        print(student_id, "has multiple logins without logout")


"""
OUTPUT:

----- STUDENT ACTIVITY REPORT -----

Student ID: S101
Name: Asha
Total Logins: 2
Total Submissions: 0

Student ID: S102
Name: Ravi
Total Logins: 0
Total Submissions: 1

----- DAILY ACTIVITY STATISTICS -----
2025-03-10 : 3 activities
2025-03-11 : 1 activities

----- ABNORMAL BEHAVIOR -----
S101 has multiple logins without logout
"""
