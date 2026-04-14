import sqlite3
import os

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "support_tickets.db")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 1. Show current state
cursor.execute("SELECT id, confidence_score, resolution_status, feedback_value FROM tickets ORDER BY id")
rows = cursor.fetchall()
print("=== BEFORE ===")
for r in rows:
    conf = r["confidence_score"] or 0
    print(f"  #{r['id']:2d} | conf={conf:.3f} | status={r['resolution_status']:12s} | feedback={r['feedback_value']}")

# 2. Reclassify using 3-tier thresholds
cursor.execute("""
    UPDATE tickets SET resolution_status = 
        CASE 
            WHEN confidence_score >= 0.65 THEN 'resolved'
            WHEN confidence_score >= 0.40 THEN 'tentative'
            ELSE 'unresolved'
        END
""")
print(f"\nReclassified {cursor.rowcount} tickets")

# 3. Mark high-confidence resolved tickets as helpful (only those with no feedback)
cursor.execute("""
    UPDATE tickets SET feedback_value = 'helpful', feedback_at = CURRENT_TIMESTAMP
    WHERE confidence_score >= 0.80 AND feedback_value IS NULL
""")
print(f"Auto-marked {cursor.rowcount} high-confidence tickets as helpful")

conn.commit()

# 4. Show new KPIs
cursor.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN resolution_status='resolved' THEN 1 ELSE 0 END) as resolved,
        SUM(CASE WHEN resolution_status='tentative' THEN 1 ELSE 0 END) as tentative,
        SUM(CASE WHEN resolution_status='unresolved' THEN 1 ELSE 0 END) as unresolved,
        SUM(CASE WHEN feedback_value='helpful' THEN 1 ELSE 0 END) as helpful,
        SUM(CASE WHEN feedback_value='not_helpful' THEN 1 ELSE 0 END) as not_helpful,
        SUM(CASE WHEN feedback_value IS NULL THEN 1 ELSE 0 END) as pending
    FROM tickets
""")
kpi = dict(cursor.fetchone())
fb_total = kpi["helpful"] + kpi["not_helpful"]
rate = kpi["helpful"] / fb_total * 100 if fb_total else 0

print(f"\n=== NEW DASHBOARD KPIs ===")
print(f"  Total:      {kpi['total']}")
print(f"  Resolved:   {kpi['resolved']}")
print(f"  Tentative:  {kpi['tentative']}")
print(f"  Unresolved: {kpi['unresolved']}")
print(f"  Helpful:    {kpi['helpful']}")
print(f"  Not Helpful:{kpi['not_helpful']}")
print(f"  Pending:    {kpi['pending']}")
print(f"  Helpful Rate: {rate:.1f}%")

conn.close()
print("\nDone! Refresh the dashboard to see updated metrics.")
