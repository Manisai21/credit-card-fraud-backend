import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def send_email(content):
    try:
        # Set up the server using environment variables
        smtp_server = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_user = os.getenv("SMTP_USER", "cursor1199@gmail.com")  # Use the environment variable
        smtp_password = os.getenv("SMTP_PASSWORD", "bpgg apkp qpso syam")  # Use the environment variable

        # Create the email
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = os.getenv("ADMIN_EMAIL", "default@example.com")  # Use the environment variable for the recipient
        msg['Subject'] = content['subject']

        # Attach the email body
        msg.attach(MIMEText(content['body'], 'plain'))

        # Connect to the server and send the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")