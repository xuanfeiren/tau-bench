#!/usr/bin/env python3
"""
Gemini High Throughput Test - 2000 Requests Per Minute

This script tests whether you can achieve 2000 requests per minute with Gemini models.
It performs controlled burst testing and sustained rate testing to identify actual limits.

Test scenarios:
1. Burst test: 2000 requests as fast as possible
2. Sustained rate: 2000 requests over 60 seconds (33.3 req/sec)
3. Progressive scaling: Start slow and ramp up to find breaking point

Requirements:
- litellm package
- GEMINI_API_KEY environment variable set

Usage:
    python gemini_high_throughput_test.py
"""

import os
import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import litellm

# Suppress LiteLLM debug info for cleaner output
litellm.suppress_debug_info = True

@dataclass
class TestResult:
    """Container for test results"""
    test_name: str
    total_requests: int
    successful: int
    rate_limited: int
    failed: int
    duration: float
    requests_per_second: float
    avg_response_time: float
    first_rate_limit_at: Optional[int] = None
    errors: List[str] = None

class HighThroughputTester:
    def __init__(self):
        self.results = []
        
    async def single_request(self, model: str, request_id: int, model_type: str = "text") -> Dict[str, Any]:
        """Make a single request and return result info"""
        start_time = time.time()
        
        try:
            if model_type == "text":
                response = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": f"Test {request_id}"}],
                    max_tokens=5  # Keep responses small for speed
                )
            else:  # embedding
                response = await litellm.aembedding(
                    model=model,
                    input=f"Test embedding {request_id}"
                )
            
            response_time = time.time() - start_time
            return {
                "status": "success",
                "response_time": response_time,
                "request_id": request_id
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e).lower()
            
            if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
                return {
                    "status": "rate_limited",
                    "response_time": response_time,
                    "request_id": request_id,
                    "error": "Rate limited"
                }
            else:
                return {
                    "status": "failed",
                    "response_time": response_time,
                    "request_id": request_id,
                    "error": str(e)[:100]
                }

    async def burst_test(self, model: str, target_requests: int = 2000, model_type: str = "text") -> TestResult:
        """Test maximum burst capacity - fire all requests at once"""
        print(f"🚀 BURST TEST: {target_requests} requests as fast as possible")
        print(f"   Model: {model}")
        print(f"   Starting burst test...")
        
        start_time = time.time()
        
        # Create all tasks at once
        tasks = [
            self.single_request(model, i, model_type) 
            for i in range(target_requests)
        ]
        
        # Execute all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze results
        successful = 0
        rate_limited = 0
        failed = 0
        response_times = []
        first_rate_limit_at = None
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed += 1
                errors.append(f"Task {i}: {str(result)[:50]}")
            elif isinstance(result, dict):
                if result["status"] == "success":
                    successful += 1
                    response_times.append(result["response_time"])
                elif result["status"] == "rate_limited":
                    rate_limited += 1
                    if first_rate_limit_at is None:
                        first_rate_limit_at = result["request_id"]
                else:
                    failed += 1
                    errors.append(result.get("error", "Unknown error"))
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        requests_per_second = target_requests / duration
        
        print(f"   ✅ Completed in {duration:.2f} seconds")
        print(f"   📊 Results: {successful} success, {rate_limited} rate limited, {failed} failed")
        print(f"   ⚡ Rate: {requests_per_second:.1f} requests/second")
        
        return TestResult(
            test_name="Burst Test",
            total_requests=target_requests,
            successful=successful,
            rate_limited=rate_limited,
            failed=failed,
            duration=duration,
            requests_per_second=requests_per_second,
            avg_response_time=avg_response_time,
            first_rate_limit_at=first_rate_limit_at,
            errors=errors[:10]  # Keep only first 10 errors
        )

    async def sustained_rate_test(self, model: str, target_requests: int = 2000, duration_seconds: int = 60, model_type: str = "text") -> TestResult:
        """Test sustained rate - spread requests evenly over time period"""
        target_rate = target_requests / duration_seconds
        print(f"⏱️  SUSTAINED RATE TEST: {target_requests} requests over {duration_seconds} seconds")
        print(f"   Model: {model}")
        print(f"   Target rate: {target_rate:.1f} requests/second")
        
        start_time = time.time()
        
        successful = 0
        rate_limited = 0
        failed = 0
        response_times = []
        first_rate_limit_at = None
        errors = []
        
        # Calculate delay between requests
        delay = duration_seconds / target_requests
        
        async def scheduled_request(request_id: int, scheduled_time: float):
            # Wait until scheduled time
            current_time = time.time()
            if scheduled_time > current_time:
                await asyncio.sleep(scheduled_time - current_time)
            
            return await self.single_request(model, request_id, model_type)
        
        # Schedule all requests
        tasks = []
        for i in range(target_requests):
            scheduled_time = start_time + (i * delay)
            task = scheduled_request(i, scheduled_time)
            tasks.append(task)
        
        # Execute with progress updates
        batch_size = 100
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            # Process batch results
            for j, result in enumerate(batch_results):
                request_id = i + j
                
                if isinstance(result, Exception):
                    failed += 1
                    errors.append(f"Request {request_id}: {str(result)[:50]}")
                elif isinstance(result, dict):
                    if result["status"] == "success":
                        successful += 1
                        response_times.append(result["response_time"])
                    elif result["status"] == "rate_limited":
                        rate_limited += 1
                        if first_rate_limit_at is None:
                            first_rate_limit_at = request_id
                    else:
                        failed += 1
                        errors.append(result.get("error", "Unknown error"))
            
            # Progress update
            completed = min(i + batch_size, target_requests)
            elapsed = time.time() - start_time
            current_rate = completed / elapsed if elapsed > 0 else 0
            print(f"   Progress: {completed}/{target_requests} ({completed/target_requests*100:.1f}%) - Rate: {current_rate:.1f} req/s")
        
        total_duration = time.time() - start_time
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        actual_rate = target_requests / total_duration
        
        print(f"   ✅ Completed in {total_duration:.2f} seconds")
        print(f"   📊 Results: {successful} success, {rate_limited} rate limited, {failed} failed")
        print(f"   ⚡ Actual rate: {actual_rate:.1f} requests/second")
        
        return TestResult(
            test_name="Sustained Rate Test",
            total_requests=target_requests,
            successful=successful,
            rate_limited=rate_limited,
            failed=failed,
            duration=total_duration,
            requests_per_second=actual_rate,
            avg_response_time=avg_response_time,
            first_rate_limit_at=first_rate_limit_at,
            errors=errors[:10]
        )

    async def progressive_scaling_test(self, model: str, model_type: str = "text") -> List[TestResult]:
        """Test progressive scaling to find the breaking point"""
        print(f"📈 PROGRESSIVE SCALING TEST: Find the breaking point")
        print(f"   Model: {model}")
        
        # Test different request volumes
        test_volumes = [50, 100, 200, 500, 1000, 1500, 2000]
        results = []
        
        for volume in test_volumes:
            print(f"\n   Testing {volume} requests...")
            
            # Quick burst test for this volume
            start_time = time.time()
            
            tasks = [self.single_request(model, i, model_type) for i in range(volume)]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            duration = time.time() - start_time
            
            # Analyze results
            successful = sum(1 for r in batch_results if isinstance(r, dict) and r["status"] == "success")
            rate_limited = sum(1 for r in batch_results if isinstance(r, dict) and r["status"] == "rate_limited")
            failed = sum(1 for r in batch_results if isinstance(r, Exception) or (isinstance(r, dict) and r["status"] == "failed"))
            
            response_times = [r["response_time"] for r in batch_results if isinstance(r, dict) and r["status"] == "success"]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            requests_per_second = volume / duration
            
            result = TestResult(
                test_name=f"Progressive Test ({volume} requests)",
                total_requests=volume,
                successful=successful,
                rate_limited=rate_limited,
                failed=failed,
                duration=duration,
                requests_per_second=requests_per_second,
                avg_response_time=avg_response_time
            )
            
            results.append(result)
            
            print(f"      Results: {successful}/{volume} successful ({successful/volume*100:.1f}%)")
            print(f"      Rate limited: {rate_limited}")
            print(f"      Rate: {requests_per_second:.1f} req/s")
            
            # If we hit significant rate limiting, we found the limit
            if rate_limited > volume * 0.1:  # More than 10% rate limited
                print(f"      🚫 Hit significant rate limiting at {volume} requests")
                break
            
            # Brief pause between tests
            await asyncio.sleep(2)
        
        return results

    def generate_report(self):
        """Generate comprehensive report of all test results"""
        print("\n" + "=" * 80)
        print("📋 HIGH THROUGHPUT TEST REPORT - 2000 REQUESTS PER MINUTE")
        print("=" * 80)
        
        if not self.results:
            print("No test results available!")
            return
        
        # Summary table
        print(f"\n📊 Test Results Summary:")
        print(f"{'Test Name':<25} {'Requests':<10} {'Success':<8} {'Rate Lim':<9} {'Failed':<7} {'Rate (req/s)':<12} {'Avg Time':<10}")
        print("-" * 80)
        
        for result in self.results:
            success_rate = result.successful / result.total_requests * 100
            print(f"{result.test_name:<25} {result.total_requests:<10} {result.successful:<8} {result.rate_limited:<9} {result.failed:<7} {result.requests_per_second:<12.1f} {result.avg_response_time:<10.3f}s")
        
        # Analysis
        print(f"\n🔍 Analysis:")
        
        # Find best performance
        best_sustained_rate = 0
        best_burst_rate = 0
        
        for result in self.results:
            if "Sustained" in result.test_name and result.rate_limited == 0:
                best_sustained_rate = max(best_sustained_rate, result.requests_per_second)
            if "Burst" in result.test_name:
                best_burst_rate = max(best_burst_rate, result.requests_per_second)
        
        print(f"   • Best sustained rate (no rate limits): {best_sustained_rate:.1f} requests/second")
        print(f"   • Best burst rate: {best_burst_rate:.1f} requests/second")
        
        # 2000 requests/minute analysis
        target_rate_per_second = 2000 / 60  # 33.33 req/s
        print(f"   • Target rate (2000/min): {target_rate_per_second:.1f} requests/second")
        
        can_achieve_target = any(
            result.requests_per_second >= target_rate_per_second and result.rate_limited == 0 
            for result in self.results
        )
        
        if can_achieve_target:
            print(f"   ✅ CAN achieve 2000 requests per minute!")
        else:
            print(f"   ❌ CANNOT achieve 2000 requests per minute without rate limiting")
        
        # Rate limiting analysis
        rate_limited_tests = [r for r in self.results if r.rate_limited > 0]
        if rate_limited_tests:
            print(f"\n🚫 Rate Limiting Detected:")
            for result in rate_limited_tests:
                if result.first_rate_limit_at is not None:
                    print(f"   • {result.test_name}: First rate limit at request #{result.first_rate_limit_at}")
                else:
                    print(f"   • {result.test_name}: {result.rate_limited} requests rate limited")
        
        # Save detailed report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "test_results": [
                {
                    "test_name": r.test_name,
                    "total_requests": r.total_requests,
                    "successful": r.successful,
                    "rate_limited": r.rate_limited,
                    "failed": r.failed,
                    "duration": r.duration,
                    "requests_per_second": r.requests_per_second,
                    "avg_response_time": r.avg_response_time,
                    "first_rate_limit_at": r.first_rate_limit_at,
                    "errors": r.errors
                }
                for r in self.results
            ],
            "analysis": {
                "can_achieve_2000_per_minute": can_achieve_target,
                "best_sustained_rate": best_sustained_rate,
                "best_burst_rate": best_burst_rate,
                "target_rate_per_second": target_rate_per_second
            }
        }
        
        with open("/Users/xuanfeiren/Documents/tau-bench/my_test_api/high_throughput_report.json", "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: high_throughput_report.json")

async def main():
    """Main function to run high throughput tests"""
    print("🚀 GEMINI HIGH THROUGHPUT TEST - 2000 REQUESTS PER MINUTE")
    print("=" * 80)
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  ERROR: GEMINI_API_KEY environment variable not set!")
        print("   Set your API key: export GEMINI_API_KEY='your-key-here'")
        return
    
    # Test configuration
    models_to_test = [
        ("gemini/gemini-2.0-flash", "text"),
        ("gemini/text-embedding-004", "embedding")
    ]
    
    tester = HighThroughputTester()
    
    for model, model_type in models_to_test:
        print(f"\n🎯 Testing model: {model} ({model_type})")
        print("=" * 60)
        
        try:
            # Test 1: Progressive scaling to find limits
            print(f"\n1️⃣  Progressive Scaling Test")
            progressive_results = await tester.progressive_scaling_test(model, model_type)
            tester.results.extend(progressive_results)
            
            # Test 2: Sustained rate test (2000 requests over 60 seconds)
            print(f"\n2️⃣  Sustained Rate Test (2000 requests in 60 seconds)")
            sustained_result = await tester.sustained_rate_test(model, 2000, 60, model_type)
            tester.results.append(sustained_result)
            
            # Test 3: Burst test (2000 requests as fast as possible)
            print(f"\n3️⃣  Burst Test (2000 requests maximum speed)")
            burst_result = await tester.burst_test(model, 2000, model_type)
            tester.results.append(burst_result)
            
        except Exception as e:
            print(f"❌ Error testing {model}: {e}")
    
    # Generate final report
    tester.generate_report()
    
    print(f"\n🎉 High throughput testing completed!")

if __name__ == "__main__":
    asyncio.run(main())
