#!/usr/bin/env python3
"""
Simple script to test Gemini embedding API rate limits with async calls.
Tests gemini-embedding-001 model with concurrent requests.
"""

import asyncio
import time
from datetime import datetime
import litellm
from collections import defaultdict

# Configure litellm
litellm.set_verbose = False

class RateLimitTester:
    def __init__(self, model_name):
        self.model_name = model_name
        self.results = defaultdict(int)
        self.start_time = None
        self.end_time = None

    async def call_embedding(self, i):
        """Make a single embedding API call"""
        try:
            start = time.time()
            response = await litellm.aembedding(
                model=self.model_name,
                input=f"Test embedding request #{i}"
            )
            duration = time.time() - start

            self.results['success'] += 1
            print(f"✅ Request {i:4d} - Success ({duration:.2f}s)")
            return True

        except Exception as e:
            error_str = str(e).lower()

            if "429" in str(e) or "rate limit" in error_str or "quota" in error_str:
                self.results['rate_limited'] += 1
                print(f"🚫 Request {i:4d} - Rate Limited")
            else:
                self.results['error'] += 1
                print(f"❌ Request {i:4d} - Error: {type(e).__name__}: {str(e)[:100]}")

            return False

    async def run_test(self, num_requests=50, batch_size=10, delay_between_batches=0.1):
        """
        Run the rate limit test with batched async requests.

        Args:
            num_requests: Total number of requests to make
            batch_size: Number of concurrent requests per batch
            delay_between_batches: Seconds to wait between batches
        """
        print(f"\n{'='*70}")
        print(f"🚀 Testing {self.model_name}")
        print(f"{'='*70}")
        print(f"Total requests: {num_requests}")
        print(f"Batch size: {batch_size}")
        print(f"Delay between batches: {delay_between_batches}s")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        self.start_time = time.time()

        # Process requests in batches
        for batch_start in range(0, num_requests, batch_size):
            batch_end = min(batch_start + batch_size, num_requests)
            batch_requests = range(batch_start, batch_end)

            # Run batch concurrently
            tasks = [self.call_embedding(i) for i in batch_requests]
            await asyncio.gather(*tasks)

            # Small delay between batches to avoid overwhelming the API
            if batch_end < num_requests:
                await asyncio.sleep(delay_between_batches)

        self.end_time = time.time()
        self.print_summary()

    def print_summary(self):
        """Print test results summary"""
        total_time = self.end_time - self.start_time
        total_requests = sum(self.results.values())

        print(f"\n{'='*70}")
        print(f"📊 Test Summary for {self.model_name}")
        print(f"{'='*70}")
        print(f"✅ Successful:     {self.results['success']:5d} ({self.results['success']/total_requests*100:.1f}%)")
        print(f"🚫 Rate Limited:   {self.results['rate_limited']:5d} ({self.results['rate_limited']/total_requests*100:.1f}%)")
        print(f"❌ Errors:         {self.results['error']:5d} ({self.results['error']/total_requests*100:.1f}%)")
        print(f"{'─'*70}")
        print(f"Total Requests:    {total_requests:5d}")
        print(f"Total Time:        {total_time:.2f}s")
        print(f"Avg Time/Request:  {total_time/total_requests:.3f}s")
        print(f"Requests/Second:   {total_requests/total_time:.2f}")
        print(f"{'='*70}\n")


async def main():
    """Main test runner"""
    # Test configuration
    model = "gemini/gemini-embedding-001"

    tester = RateLimitTester(model)

    # Adjust these parameters to test different scenarios:
    # - num_requests: Total number of API calls
    # - batch_size: How many concurrent requests (higher = more aggressive)
    # - delay_between_batches: Wait time between batches (lower = more aggressive)

    await tester.run_test(
        num_requests=100,      # Total requests to make
        batch_size=2,          # Concurrent requests per batch
        delay_between_batches=1.0  # Seconds between batches
    )

    print("\n💡 Tips:")
    print("  - Increase batch_size for more aggressive testing")
    print("  - Decrease delay_between_batches to test faster")
    print("  - Watch for rate limit responses to find your limits")


if __name__ == "__main__":
    asyncio.run(main())
