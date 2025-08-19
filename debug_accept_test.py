#!/usr/bin/env python3
"""
Simple test to see if models understand the accept action format.
"""
import json

# Test the exact prompt structure our models are receiving
test_prompt = '''You are an expert negotiator in a car sale negotiation.

CURRENT SITUATION:
- You are the buyer
- Seller just offered €42,000
- Your BATNA (alternative) is €44,000
- This is a PROFITABLE offer for you (saves €2,000)

CRITICAL: The seller offered €42,000 which is BELOW your BATNA of €44,000.
This is a PROFITABLE deal - you save €2,000!

INSTRUCTION: Since this is profitable, you must accept it immediately.

RESPONSE FORMAT: Respond with ONLY a JSON object:
- To accept: {"type": "accept"}
- To make counter-offer: {"type": "offer", "price": 41000}

What is your response?
'''

print("=== TEST PROMPT ===")
print(test_prompt)
print("\n=== EXPECTED RESPONSE ===")
print('{"type": "accept"}')
print("\nThis tests if the model understands:")
print("1. That €42,000 < €44,000 BATNA = profitable")  
print("2. That profitable offers should be accepted")
print("3. The JSON format for acceptance")
