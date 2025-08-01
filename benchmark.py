"""
Benchmark the API response time to ensure it meets the 20-second target.
Run this script to test the optimized version against a set of test questions.
"""

import requests
import time
import json
import os
import statistics
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = os.getenv("API_URL", "http://localhost:8000")
API_TOKEN = os.getenv("HACKRX_API_TOKEN", "supersecrettoken")
TEST_DOC_URL = "https://example.com/sample_insurance_policy.pdf"  # Change to your test document URL

# Test questions
TEST_QUESTIONS = [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?",
    "Are health check-ups covered?",
    "What defines a hospital under this policy?",
    "Are AYUSH treatments covered?",
    "Is maternity covered?"
]

def run_benchmark(num_runs=3):
    """Run the benchmark multiple times and calculate average response time"""
    all_times = []
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Warm-up run (not counted in stats)
    print("Performing warm-up request...")
    payload = {
        "documents": TEST_DOC_URL,
        "questions": TEST_QUESTIONS[:1]
    }
    requests.post(f"{BASE_URL}/hackrx/run", headers=headers, json=payload)
    
    # Benchmark runs
    print(f"Running {num_runs} benchmark iterations...")
    
    for i in range(num_runs):
        print(f"\nIteration {i+1}/{num_runs}:")
        
        payload = {
            "documents": TEST_DOC_URL,
            "questions": TEST_QUESTIONS
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=headers, json=payload)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        all_times.append(elapsed_time)
        
        print(f"  Response time: {elapsed_time:.2f} seconds")
        print(f"  Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Received {len(data['answers'])} answers")
        else:
            print(f"  Error: {response.text}")
    
    # Calculate statistics
    avg_time = statistics.mean(all_times)
    min_time = min(all_times)
    max_time = max(all_times)
    std_dev = statistics.stdev(all_times) if len(all_times) > 1 else 0
    
    print("\nBenchmark Results:")
    print(f"  Average response time: {avg_time:.2f} seconds")
    print(f"  Min response time: {min_time:.2f} seconds")
    print(f"  Max response time: {max_time:.2f} seconds")
    print(f"  Standard deviation: {std_dev:.2f} seconds")
    
    if avg_time <= 20.0:
        print("\n✅ Performance target met! Average response time <= 20 seconds")
    else:
        print("\n⚠️ Performance target not met. Average response time > 20 seconds")

if __name__ == "__main__":
    print("API Performance Benchmark")
    print("=" * 30)
    run_benchmark()
