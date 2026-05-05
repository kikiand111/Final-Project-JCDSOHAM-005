# Hotel Booking System

A structured and scalable Hotel Booking Management System designed to handle hotel listings, room availability, customer reservations, and payment processing through a clean relational data model.

---

## Project Overview

This project is a backend-oriented system that simulates a real-world hotel booking platform. It focuses on data modeling, relational database design, and booking workflow logic to ensure efficient reservation handling.

The system is designed to be:
- Scalable
- Modular
- Easy to extend for production-level features

---

## Objectives

- Design a normalized relational database for hotel booking
- Manage users, hotels, rooms, bookings, and payments
- Implement a clear booking lifecycle
- Ensure data consistency and integrity across entities
- Provide a foundation for a production-grade booking system

---

## System Design and Data Modeling

### Core Entities

#### Users
Stores customer information.

- id_user (int, primary key)
- name (string)
- email (string, unique)
- phone (string)

---

#### Hotels
Represents hotel properties.

- id_hotel (int, primary key)
- name (string)
- location (string)
- rating (float)

---

#### Rooms
Represents available hotel rooms.

- id_room (int, primary key)
- id_hotel (int, foreign key)
- room_type (string)
- price (decimal)
- status (string: Available / Booked)

---

#### Bookings
Handles reservation transactions.

- id_booking (int, primary key)
- id_user (int, foreign key)
- id_room (int, foreign key)
- check_in (date)
- check_out (date)
- status (string: Pending / Confirmed / Completed)

---

#### Payments
Handles booking payments.

- id_payment (int, primary key)
- id_booking (int, foreign key)
- method (string)
- total_amount (decimal)
- status (string: Paid / Unpaid)

---

## Database Relationships

- User has many Bookings
- Hotel has many Rooms
- Room has many Bookings
- Booking has one Payment

---

## System Workflow

1. User registers or logs in
2. User browses available hotels
3. User selects a room
4. System checks room availability
5. Booking is created with status Pending
6. User completes payment
7. System validates payment
8. Booking status is updated to Confirmed
9. After checkout, status becomes Completed

---

## Architecture

Client / API Consumer  
→ Controller Layer (API Routes)  
→ Service Layer (Business Logic)  
→ Database Layer (Relational Database)

---

## Features

- User management system
- Hotel and room listing
- Room availability tracking
- Booking lifecycle management
- Payment tracking system
- Structured relational database design

---

## Tech Stack (Suggested)

- Database: MySQL / PostgreSQL
- Backend: Node.js / Python (Flask / Django)
- API: RESTful API
- Design: Entity Relationship Diagram (ERD)

---

## ERD Overview

User ───< Booking >─── Room ───< Hotel  
             │  
             └──── Payment  

---

## Future Improvements

- JWT authentication and authorization
- Admin dashboard for hotel management
- Real-time room availability updates
- Payment gateway integration
- Review and rating system
- Discount and promotion system

---

## Project Status

- Core system design completed
- Database modeling completed
- Ready for backend implementation
- Optional frontend integration available

---

## License

This project is for educational and portfolio purposes.

---

## Author

Developed as part of a structured system design and database modeling learning project.
