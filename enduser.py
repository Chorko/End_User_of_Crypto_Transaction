from oldfunc import EthereumAddressClusterer
import requests
import time
import os
import numpy as np
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

class AddressFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/api"
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 200ms between requests

    def _make_api_request(self, params: Dict) -> Dict:
        """
        Make API request with rate limiting and error handling
        
        Args:
            params (Dict): API request parameters
        
        Returns:
            Dict: JSON response from API or None if request fails
        """
        try:
            # Ensure rate limiting
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last_request)
            
            params['apikey'] = self.api_key
            
            response = requests.get(self.base_url, params=params)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == '1' or 'result' in result:
                    return result
                elif result.get('message') == 'NOTOK' and 'Invalid API Key' in result.get('result', ''):
                    raise Exception("Invalid API Key. Please check your Etherscan API key.")
                else:
                    print(f"API Error: {result.get('message', 'Unknown error')}")
                    return None
            elif response.status_code == 429:
                print("Rate limit exceeded. Waiting before retrying...")
                time.sleep(1)  # Wait longer on rate limit
                return self._make_api_request(params)  # Retry
            else:
                print(f"HTTP Error: {response.status_code}")
                return None
            
        except Exception as e:
            print(f"Request Error: {str(e)}")
            return None

    def get_exchange_addresses(self, count: int = 10) -> List[str]:
        """Get addresses from major exchange transactions"""
        # Known exchange addresses to start with
        exchanges = [
            "0x28C6c06298d514Db089934071355E5743bf21d60",  # Binance
            "0x2910543af39aba0cd09dbb2d50200b3e800a63d2"   # Kraken
        ]
        
        addresses = set()
        for exchange in exchanges:
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': exchange,
                'startblock': 'latest',
                'page': 1,
                'offset': 20,
                'sort': 'desc'
            }
            
            response = self._make_api_request(params)
            if response and 'result' in response:
                for tx in response['result']:
                    addresses.add(tx['from'])
                    addresses.add(tx['to'])
                    if len(addresses) >= count:
                        break
        
        return list(addresses)[:count]

    def get_defi_addresses(self, count: int = 10) -> List[str]:
        """Get addresses interacting with DeFi protocols"""
        # Popular DeFi contract addresses
        defi_contracts = [
            "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",  # Aave
            "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45"   # Uniswap
        ]
        
        addresses = set()
        for contract in defi_contracts:
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': contract,
                'startblock': 'latest',
                'page': 1,
                'offset': 20,
                'sort': 'desc'
            }
            
            response = self._make_api_request(params)
            if response and 'result' in response:
                for tx in response['result']:
                    addresses.add(tx['from'])
                    if len(addresses) >= count:
                        break
        
        return list(addresses)[:count]

    def get_nft_addresses(self, count: int = 10) -> List[str]:
        """Get addresses from NFT marketplace transactions"""
        # Popular NFT marketplace contracts
        nft_contracts = [
            "0x7Be8076f4EA4A4AD08075C2508e481d6C946D12b",  # OpenSea
            "0x74312363e45dcaba76c59ec49a7aa8a65a67eed3"   # X2Y2
        ]
        
        addresses = set()
        for contract in nft_contracts:
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': contract,
                'startblock': 'latest',
                'page': 1,
                'offset': 20,
                'sort': 'desc'
            }
            
            response = self._make_api_request(params)
            if response and 'result' in response:
                for tx in response['result']:
                    addresses.add(tx['from'])
                    if len(addresses) >= count:
                        break
        
        return list(addresses)[:count]

    def get_individual_addresses(self, count: int = 10) -> List[str]:
        """Get addresses from recent normal transactions"""
        # Get transactions from a recent block
        params = {
            'module': 'proxy',  # Using proxy API for latest blocks
            'action': 'eth_getBlockByNumber',
            'tag': 'latest',
            'boolean': 'true'
        }
        
        addresses = set()
        response = self._make_api_request(params)
        
        if response and 'result' in response and 'transactions' in response['result']:
            for tx in response['result']['transactions']:
                # Filter out contract creations and known patterns
                if 'to' in tx and tx['to'] and not tx['to'].startswith('0x000000'):
                    addresses.add(tx['from'].lower())
                    if len(addresses) >= count:
                        break
        
        # If we need more addresses, fetch from normal transactions
        if len(addresses) < count:
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': '0x0000000000000000000000000000000000000000',  # Use null address as reference
                'startblock': 'latest',
                'page': 1,
                'offset': 50,
                'sort': 'desc'
            }
            
            response = self._make_api_request(params)
            if response and 'result' in response:
                for tx in response['result']:
                    if 'to' in tx and tx['to'] and not tx['to'].startswith('0x000000'):
                        addresses.add(tx['from'].lower())
                        if len(addresses) >= count:
                            break
        
        return list(addresses)[:count]

class EndUserPredictor:
    def __init__(self, transaction_graph, clusters):
        self.transaction_graph = transaction_graph
        self.clusters = clusters
        self.isolation_forest = None
        self.random_forest = None
        self.scaler = StandardScaler()
        
    def extract_features(self, addresses: List[str]) -> np.ndarray:
        """Extract numerical features for each address"""
        features = []
        for address in addresses:
            if address in self.transaction_graph:
                # Node-level features
                degree = self.transaction_graph.degree(address)
                in_degree = self.transaction_graph.in_degree(address)
                out_degree = self.transaction_graph.out_degree(address)
                
                # Get cluster information
                cluster_id = -1
                for cid, cluster_addrs in self.clusters.items():
                    if address in cluster_addrs:
                        cluster_id = cid
                        break
                
                # Create feature vector
                feature_vector = [
                    degree,
                    in_degree,
                    out_degree,
                    cluster_id,
                    len(list(self.transaction_graph.neighbors(address)))
                ]
                features.append(feature_vector)
            else:
                features.append([0, 0, 0, -1, 0])  # Default features for unknown addresses
                
        return np.array(features)
    
    def train_isolation_forest(self, addresses: List[str]):
        """Train Isolation Forest for anomaly detection"""
        features = self.extract_features(addresses)
        features_scaled = self.scaler.fit_transform(features)
        
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        self.isolation_forest.fit(features_scaled)
    
    def train_random_forest(self, addresses: List[str], labels: List[int]):
        """Train Random Forest for user classification"""
        features = self.extract_features(addresses)
        features_scaled = self.scaler.fit_transform(features)
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42
        )
        
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.random_forest.fit(X_train, y_train)
        
    def predict_user_type(self, address: str) -> Dict[str, any]:
        """Combine all models to predict user type and characteristics"""
        features = self.extract_features([address])
        features_scaled = self.scaler.transform(features)
        
        results = {
            "address": address,
            "is_anomaly": False,
            "user_category": None,
            "confidence": 0.0
        }
        
        # Anomaly detection
        if self.isolation_forest is not None:
            anomaly_score = self.isolation_forest.score_samples(features_scaled)[0]
            results["is_anomaly"] = anomaly_score < -0.5  # Threshold can be adjusted
            results["anomaly_score"] = anomaly_score
            
        # User category prediction
        if self.random_forest is not None:
            prediction = self.random_forest.predict(features_scaled)[0]
            probabilities = self.random_forest.predict_proba(features_scaled)[0]
            results["user_category"] = prediction
            results["confidence"] = max(probabilities)
            
        return results

def main():
    """
    Main function to demonstrate the usage of EthereumAddressClusterer
    with dynamically fetched addresses and end user prediction
    """
    
    # Get API key from environment variable
    api_key = os.getenv('ETHERSCAN_API_KEY')
    if not api_key:
        print("Error: ETHERSCAN_API_KEY environment variable not set")
        print("Please create a .env file with your Etherscan API key:")
        print("ETHERSCAN_API_KEY=your_api_key_here")
        return
    
    try:
        # Test API key validity first
        fetcher = AddressFetcher(api_key)
        test_params = {
            'module': 'account',
            'action': 'balance',
            'address': '0x0000000000000000000000000000000000000000'
        }
        if not fetcher._make_api_request(test_params):
            print("Failed to validate API key. Please check your Etherscan API key.")
            return
            
        print("API key validated successfully!")
        
        # Initialize clusterer
        clusterer = EthereumAddressClusterer(api_key)
        
        # Fetch addresses from different categories
        print("\nFetching addresses from recent transactions...")
        
        addresses = {
            'exchanges': fetcher.get_exchange_addresses(5),  # Reduced count for testing
            'defi': fetcher.get_defi_addresses(5),
            'nft': fetcher.get_nft_addresses(5),
            'individuals': fetcher.get_individual_addresses(5)
        }
        
        # Check if we got enough addresses
        total_addresses = sum(len(addrs) for addrs in addresses.values())
        if total_addresses == 0:
            raise Exception("Failed to fetch any addresses")
        
        # Flatten addresses and remove duplicates
        all_addresses = list(set([
            addr for category in addresses.values() 
            for addr in category if addr  # Filter out None values
        ]))
        
        print(f"\nAnalyzing {len(all_addresses)} unique addresses...")
        
        # Perform clustering analysis with more conservative parameters
        print("\nStarting clustering analysis...")
        clusters = clusterer.cluster_addresses(all_addresses, eps=0.5, min_samples=2)
        
        # Print clustering results with category labels
        print("\nClustering Results:")
        for cluster_id, cluster_addresses in clusters.items():
            if cluster_id == -1:
                print("\nNoise/Outliers (addresses that don't fit in any cluster):")
            else:
                print(f"\nCluster {cluster_id}:")
            
            for address in cluster_addresses:
                category = "Unknown"
                for cat, addrs in addresses.items():
                    if address and addrs and address.lower() in [a.lower() for a in addrs if a]:
                        category = cat.capitalize()
                        break
                print(f"  {address} ({category})")
        
        # Build and analyze transaction graph
        print("\nBuilding transaction graph...")
        clusterer.build_transaction_graph(all_addresses)
        
        # Print graph statistics
        print("\nTransaction Graph Statistics:")
        print(f"Number of unique addresses (nodes): {clusterer.transaction_graph.number_of_nodes()}")
        print(f"Number of transactions (edges): {clusterer.transaction_graph.number_of_edges()}")
        
        # After clustering analysis, initialize and train the predictor
        print("\nInitializing End User Predictor...")
        predictor = EndUserPredictor(clusterer.transaction_graph, clusters)
        
        # Create synthetic labels for demonstration (in production, you'd need real labeled data)
        address_list = list(clusterer.transaction_graph.nodes())
        synthetic_labels = np.random.randint(0, 3, size=len(address_list))  # 3 user categories
        
        print("\nTraining machine learning models...")
        predictor.train_isolation_forest(address_list)
        predictor.train_random_forest(address_list, synthetic_labels)
        
        print("\nAnalyzing sample addresses...")
        for address in address_list[:5]:  # Analyze first 5 addresses as example
            prediction = predictor.predict_user_type(address)
            print(f"\nAddress: {address}")
            print(f"Anomaly: {'Yes' if prediction['is_anomaly'] else 'No'}")
            print(f"User Category: {prediction['user_category']}")
            print(f"Confidence: {prediction['confidence']:.2f}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your API key and internet connection.")

if __name__ == "__main__":
    main() 