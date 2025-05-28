import requests
import unittest
import sys
import json
from datetime import datetime

class CryptoAPITester:
    def __init__(self, base_url="https://2001c91c-95dd-4e9b-87e6-10bf1627cb23.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.results = []

    def run_test(self, name, method, endpoint, expected_status=200, data=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers)
            
            success = response.status_code == expected_status
            
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"Response: {json.dumps(response_data, indent=2)[:500]}...")
                except:
                    print(f"Response: {response.text[:200]}...")
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"Response: {response.text[:200]}...")
            
            result = {
                "name": name,
                "success": success,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "response": response.json() if success and response.text else response.text[:200]
            }
            
            self.results.append(result)
            return success, response.json() if success and response.text else None
        
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.results.append({
                "name": name,
                "success": False,
                "error": str(e)
            })
            return False, None

    def test_health_check(self):
        """Test the health check endpoint"""
        return self.run_test("Health Check", "GET", "api/health")
    
    def test_supported_cryptos(self):
        """Test the supported cryptocurrencies endpoint"""
        return self.run_test("Supported Cryptocurrencies", "GET", "api/supported-cryptos")
    
    def test_crypto_data(self, symbol="BTC"):
        """Test getting crypto data for a specific symbol"""
        return self.run_test(f"Crypto Data for {symbol}", "GET", f"api/crypto/{symbol}")
    
    def test_crypto_prediction(self, symbol="BTC"):
        """Test getting prediction for a specific symbol"""
        return self.run_test(f"Crypto Prediction for {symbol}", "GET", f"api/predict/{symbol}")
    
    def test_prediction_history(self, symbol="BTC"):
        """Test getting prediction history for a specific symbol"""
        return self.run_test(f"Prediction History for {symbol}", "GET", f"api/predictions/history/{symbol}")
    
    def print_summary(self):
        """Print a summary of all test results"""
        print("\n" + "="*50)
        print(f"📊 SUMMARY: {self.tests_passed}/{self.tests_run} tests passed")
        print("="*50)
        
        for result in self.results:
            status = "✅ PASSED" if result.get("success") else "❌ FAILED"
            print(f"{status} - {result.get('name')}")
        
        print("="*50)
        return self.tests_passed == self.tests_run

class TestCryptoAPI(unittest.TestCase):
    def setUp(self):
        self.tester = CryptoAPITester()
    
    def test_api_endpoints(self):
        """Test all API endpoints"""
        # Test health check
        success, _ = self.tester.test_health_check()
        self.assertTrue(success, "Health check failed")
        
        # Test supported cryptos
        success, cryptos = self.tester.test_supported_cryptos()
        self.assertTrue(success, "Supported cryptos endpoint failed")
        
        # Test crypto data for BTC
        success, btc_data = self.tester.test_crypto_data("BTC")
        self.assertTrue(success, "BTC data endpoint failed")
        
        # Test crypto data for ETH
        success, eth_data = self.tester.test_crypto_data("ETH")
        self.assertTrue(success, "ETH data endpoint failed")
        
        # Test prediction for BTC
        success, btc_prediction = self.tester.test_crypto_prediction("BTC")
        self.assertTrue(success, "BTC prediction endpoint failed")
        
        # Test prediction history for BTC
        success, history = self.tester.test_prediction_history("BTC")
        self.assertTrue(success, "BTC prediction history endpoint failed")
        
        # Print summary
        self.tester.print_summary()

def main():
    # Create tester
    tester = CryptoAPITester()
    
    # Run individual tests
    tester.test_health_check()
    tester.test_supported_cryptos()
    
    # Test multiple cryptocurrencies
    for symbol in ["BTC", "ETH", "ADA"]:
        tester.test_crypto_data(symbol)
        tester.test_crypto_prediction(symbol)
        tester.test_prediction_history(symbol)
    
    # Print summary
    all_passed = tester.print_summary()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    # Run as script
    sys.exit(main())
