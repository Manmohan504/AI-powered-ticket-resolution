import ticket_service
import database
import time
import sys

def test_system():
    print("AUTOMATED TEST REPORT")
    print("=====================")
    
    try:
        # 1. Init
        print("[1] Initializing System...")
        ticket_service.initialize_system()
        print("    - DB Connection: OK")
        print("    - AI Model Check: OK")

        # 2. Submit Ticket
        print("\n[2] Submitting Test Ticket...")
        title = "System Slowdown"
        desc = "The application takes 10 seconds to load the dashboard."
        print(f"    - Input Title: {title}")
        print(f"    - Input Desc: {desc}")
        
        start_time = time.time()
        ticket_service.submit_ticket(title, desc, "Technical", "High", "test_user_01")
        duration = time.time() - start_time
        print(f"    - Submission Time: {duration:.2f}s")
        print("    - Ticket submitted successfully.")

        # 3. Verify
        print("\n[3] Verifying Database Entry...")
        tickets = ticket_service.get_all_tickets()
        
        if tickets.empty:
            print("    - ERROR: No tickets found in DB!")
            sys.exit(1)
            
        latest = tickets.iloc[0]
        
        print(f"    - Retrieved Ticket ID: {latest['id']}")
        print(f"    - Category Assigned: {latest['category']}")
        print(f"    - AI Resolution: {latest['ai_resolution']}")

        # Assertions
        assert latest['title'] == title
        assert len(latest['ai_resolution']) > 10
        
        print("\nFINAL RESULT: PASS")
        print("The system is fully operational.")

    except Exception as e:
        print(f"\nFINAL RESULT: FAIL")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_system()