from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import emails
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

class ContactForm(BaseModel):
    name: str
    email: str
    phone: str
    message: str

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "manisaisaduvala21@gmail.com")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "cursor1199@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "bpgg apkp qpso syam")

@router.post("/contact/send-email")
async def send_contact_email(contact: ContactForm):
    try:
        # Create email message
        message = emails.Message(
            subject=f"New Contact Form Submission from {contact.name}",
            html=f"""
            <h2>New Contact Form Submission</h2>
            <p><strong>Name:</strong> {contact.name}</p>
            <p><strong>Email:</strong> {contact.email}</p>
            <p><strong>Phone:</strong> {contact.phone}</p>
            <h3>Message:</h3>
            <p>{contact.message}</p>
            """,
            mail_from=("Fraud Detection Contact Form", SMTP_USER)
        )

        # Send email
        response = message.send(
            to=ADMIN_EMAIL,
            smtp={
                "host": SMTP_HOST,
                "port": SMTP_PORT,
                "user": SMTP_USER,
                "password": SMTP_PASSWORD,
                "tls": True
            }
        )

        if response.status_code not in [250, 200]:
            raise HTTPException(status_code=500, detail="Failed to send email")

        return {"message": "Email sent successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))